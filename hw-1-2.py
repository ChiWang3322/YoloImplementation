import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import numpy as np
import pyrender
import cv2
from scipy.spatial.transform.rotation import Rotation
import trimesh
from jsonargparse import ArgumentParser


class CheckerboardTracker:
    def __init__(self, K_color, K_depth, side_length=4, checkerboard_pattern=(6,9), height=480, width=640, mode='renderer'):
        self.K_color = K_color
        self.K_depth = K_depth
        self.CHECKERBOARD = checkerboard_pattern
        self.side_length = side_length
        self.height = height
        self.width = width
        self.mode = mode

        # Defining the world coordinates for 3D points
        self.template_board_points3d = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
        self.template_board_points3d[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2) * self.side_length
        self.prev_img_shape = None

        self.scene = pyrender.Scene()
        self.camera = None
        self.plane_mesh = None
        self.initialized_scene = False
        self.viewer = None
        self.renderer = None

        # convert from pyrender to opencv coordinate system. To convert from pyrender to opencv we take the inverse
        self.camera_to_camera_pose = np.identity(4) # in opencv coordinate system
        self.camera_to_camera_pose[:3, :3] = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_matrix() # in pyrender coordinate system
        # define the board in trimesh
        self.plane_mesh = self.make_plane_trimesh()
        # define the camera with color intrinsics (because we use aligned depth)
        self.camera = pyrender.IntrinsicsCamera(fx=K_color[0, 0], fy=K_color[1, 1], cx=K_color[0, 2], cy=K_color[1, 2], zfar=1000)
        # light source for rendering
        light = pyrender.SpotLight(color=np.ones(3), intensity=30,
                                   innerConeAngle=np.pi / 16.0,
                                   outerConeAngle=np.pi / 6.0)
        # add the nodes to the scene
        self.scene.add(self.camera, 'camera', pose=self.camera_to_camera_pose)
        self.plane_mesh = self.scene.add(self.plane_mesh, 'checkerboard', np.identity(4))
        self.scene.add(light, pose=self.camera_to_camera_pose)

        # setup the renderer
        self.renderer = pyrender.OffscreenRenderer(width, height)
        if self.mode == 'viewer':
            # setup the viewer
            self.viewer = pyrender.Viewer(scene=self.scene, run_in_thread=True)


    def estimate_pose(self, rgb_color):
        # find corners on the pixel level.
        gray = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2GRAY)
        found_corners, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD,
                                                           flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
                                                            + cv2.CALIB_CB_NORMALIZE_IMAGE)
        # refine corners to the subpixel level.
        if found_corners:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 0.1))
            # estimate pose of the board using PnP
            found_pose, object_to_camera_rotvec, object_to_camera_tvec = cv2.solvePnP(objectPoints=self.template_board_points3d,
                                                                                      imagePoints=corners,
                                                                                      cameraMatrix=self.K_color,
                                                                                      distCoeffs=None,
                                                                                      useExtrinsicGuess=False,
                                                                                      flags=0)
            if found_pose:
                return self.make_transform(rotvec=object_to_camera_rotvec, tvec=object_to_camera_tvec), corners
            else:
                return None, corners
        else:
            return None, None

    def visualize(self, rgb_image, depth_image, detected_corners, object_to_camera_pose):
        img = rgb_image
        cv2.drawChessboardCorners(img, self.CHECKERBOARD, detected_corners, True)
        # draw board edges
        imgpoints = detected_corners[:, 0].reshape((self.CHECKERBOARD[1], self.CHECKERBOARD[0], 2))
        indices = [(0, 0), (0, 5), (8, 5), (8, 0)]
        points2d = np.array([imgpoints[i, j] for i, j in indices], dtype=np.int32)
        color = (255, 255, 0, 127)
        img = cv2.line(img, points2d[0], points2d[1], color, 3)
        img = cv2.line(img, points2d[1], points2d[2], color, 3)
        img = cv2.line(img, points2d[2], points2d[3], color, 3)
        img = cv2.line(img, points2d[3], points2d[0], color, 3)
        # overlay camera axis
        img = self.overlay_axis(img, np.identity(4))
        # overlay object axis
        img = self.overlay_axis(img, object_to_camera_pose)
        mask = np.zeros((self.height, self.width))
        mask = cv2.fillPoly(mask, [points2d], color=(255, 255, 255))
        # img[mask == 0] = [0,0,0]
        ones_mask = np.where(mask > 0, 1.0, 0.0)
        # depth to meters
        original_depth_meters = depth_image * ones_mask / 1000
        cv2.imshow('depth_image', depth_image)
        cv2.imshow("mask", mask)
        cv2.imshow("overlay", img)
        # calculate camera pose in pyrender coordinate system
        object_to_camera_pose = np.linalg.inv(self.camera_to_camera_pose) @ object_to_camera_pose
        # update visualizer pose
        if self.mode == 'viewer' and self.viewer.is_active:
            self.viewer.render_lock.acquire()
        self.scene.set_pose(self.plane_mesh, object_to_camera_pose)


        if self.mode == 'viewer' and self.viewer.is_active:
            self.viewer.render_lock.release()

        if self.mode == 'renderer':
            rendered_color, rendered_depth = self.renderer.render(self.scene)

            rendered_depth_meters = rendered_depth * ones_mask / 100
            diff = abs(original_depth_meters - rendered_depth_meters)
            rmse = (diff**2).mean()**0.5
            print(f"root mean squared error (meters) = {rmse}")
            cv2.imshow('rendered_depth', rendered_depth_meters)
            cv2.imshow('rendered_color', rendered_color)
            cv2.imshow('diff', diff)


    def make_rt_vecs(self, transform):
        rvec = Rotation.from_matrix(transform[:3,:3]).as_rotvec().reshape((3,1))
        tvec = transform[:3,3].reshape((3,1))
        return rvec, tvec

    def make_transform(self, rotvec, tvec, inv=False):
        # convert a rotation vector and translation vector to a 4x4 rigid body transformation.
        p = np.eye(4)
        rotmat = Rotation.from_rotvec(rotvec[:, 0]).as_matrix()
        tvec = tvec[:, 0]
        p[:3, :3] = rotmat
        p[:3, 3] = tvec
        if inv:
            p = np.linalg.inv(p)
        return p

    def make_plane_trimesh(self):
        # define the board mesh using trimesh
        transform_from_center = np.eye(4)
        extents = np.array([self.side_length * self.CHECKERBOARD[0], self.side_length * self.CHECKERBOARD[1], 1])
        transform_from_center[:3, 3] = np.array([0.5, 0.5, 0.5]) * extents
        transform_from_center[:3, :3] = Rotation.from_euler('xyz', (0, 0, 0), degrees=True).as_matrix()
        plane_trimesh = trimesh.primitives.Box(extents=extents, transform=transform_from_center)
        plane_mesh = pyrender.Mesh.from_trimesh(plane_trimesh)
        return plane_mesh

    def overlay_axis(self, img, pose):
        axis = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=float) * 20
        rotvec,tvec = self.make_rt_vecs(pose)
        axis2d, jac = cv2.projectPoints(axis, rotvec, tvec, self.K_color, None)
        axis2d = np.array(axis2d[:, 0], dtype=int)
        img = cv2.line(img, axis2d[0], axis2d[1], (0, 0, 255), 3)
        img = cv2.line(img, axis2d[0], axis2d[2], (0, 255, 0), 3)
        img = cv2.line(img, axis2d[0], axis2d[3], (255, 0, 0), 3)
        return img


class RealsenseSDKWrapper:
    def __init__(self, input_bagfile=None, fps=30, realtime=True, repeat_playback=True, colorizer=False):
        self.input_bagfile = input_bagfile
        self.realtime = realtime
        self.repeat_playback = repeat_playback
        self.align = rs.align(rs.stream.color)
        if colorizer:
            self.colorizer = rs.colorizer()
        else:
            self.colorizer = None

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.fps = fps
        if input_bagfile:
            rs.config.enable_device_from_file(self.config, input_bagfile, repeat_playback=self.repeat_playback)
        self.config.enable_stream(rs.stream.depth, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, rs.format.rgb8, self.fps)
        self.profile = None
        self.counter = -1

    def is_running(self):
        return self.profile is not None

    def start(self):
        self.counter = 0
        self.profile = self.pipeline.start(self.config)
        dev = self.profile.get_device()
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"Starting device with serial number: {serial}")

        mprofile = self.profile.get_stream(rs.stream.color)  # Fetch stream profile for depth stream
        cintr = mprofile.as_video_stream_profile().get_intrinsics()  # Downcast to video_stream_profile and fetch intrinsics
        self.K_color = np.array([[cintr.fx, 0, cintr.ppx],
                           [0, cintr.fy, cintr.ppy],
                           [0, 0, 1],
                           ])
        mdprofile = self.profile.get_stream(rs.stream.depth)
        dintr = mdprofile.as_video_stream_profile().get_intrinsics()
        self.height = cintr.height
        self.width = cintr.width

        self.K_depth = np.array([[dintr.fx, 0, dintr.ppx],
                                 [0, dintr.fy, dintr.ppy],
                                 [0, 0, 1]
                                 ])
        if self.input_bagfile:
            playback = self.profile.get_device().as_playback()
            playback.set_real_time(self.realtime)


    def next(self):
        self.counter += 1
        print(f"\r Frame {self.counter}", sep='', end=' ')

        frame_present, frames = self.pipeline.try_wait_for_frames()
        if not frame_present:
            raise StopIteration("No more frames received")
        # Get frame
        # depth_frame = frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = frames.get_color_frame()
        # Get aligned frames
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        if not color_frame:
            raise StopIteration("No more frames received")
        # Colorize depth frame to jet colormap
        if self.colorizer:
            aligned_depth_color_frame = self.colorizer.colorize(aligned_depth_frame)
        else:
            aligned_depth_color_frame = aligned_depth_frame

        aligned_depth_color_image = np.asanyarray(aligned_depth_color_frame.get_data())

        color_image = np.asanyarray(color_frame.get_data())
        im_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        return im_rgb, aligned_depth_color_image

    def close(self):
        self.pipeline.stop()
        print("Stopping realsense pipeline.")

    def __iter__(self):
        if not self.is_running():
            self.start()
        return self

    def __next__(self):
        return self.next()

def setup_argparser():
    parser = ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream fps and format to match the recorded.")
    parser.add_argument("-i", "--input_bagfile", type=str, help="Path to the bag file, defaults to reading data from sensor", default='Homework/HW1-2-data/20220405_220626.bag')
    parser.add_argument("-m", "--mode", choices=['viewer', 'renderer'],default='renderer', help="rendering mode")
    parser.add_argument("-f", "--fps", type=int, default=30, help="Frame rate")
    parser.add_argument("-r", "--realtime", type=bool, default=False, help="realtime recording, ignores fps")
    parser.add_argument("-l", "--repeat_playback", type=bool, default=False, help="loop/repeat the recording")
    parser.add_argument("-c", "--colorizer", type=bool, default=False, help="colorize depth")
    
    args = parser.parse_args()
    return args


def main():
    args = setup_argparser()
    reader = RealsenseSDKWrapper(input_bagfile=args.input_bagfile, fps=args.fps, realtime=args.realtime,
                                 repeat_playback=args.repeat_playback, colorizer=args.colorizer)

    reader.start()
    tracker = CheckerboardTracker(K_color=reader.K_color, K_depth=reader.K_depth, side_length=4, checkerboard_pattern=(6,9),
                                  height=reader.height, width=reader.width, mode=args.mode)
    for (rgb_color, depth) in reader:
        object_to_camera_pose, checkerboard_corners2d = tracker.estimate_pose(rgb_color=rgb_color)
        if object_to_camera_pose is not None and checkerboard_corners2d is not None:
            tracker.visualize(rgb_image=rgb_color, depth_image=depth,
                              detected_corners=checkerboard_corners2d, object_to_camera_pose=object_to_camera_pose)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    reader.close()


if __name__ == '__main__':
    main()
