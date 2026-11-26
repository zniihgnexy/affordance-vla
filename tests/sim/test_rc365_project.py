import numpy as np

from affordance_vla.sim.rc365_project import camera_from, project, visible


def test_a_point_on_the_optical_axis_projects_to_the_image_centre():
    cam = {"pos": np.zeros(3), "rot": np.eye(3), "size": 100, "focal": 80.0}
    uv, depth = project(cam, np.array([[0.0, 0.0, -2.0]]))
    assert np.allclose(uv, [[50, 50]]) and np.allclose(depth, [2.0])


def test_up_in_the_camera_is_up_in_the_image_and_right_is_right():
    cam = {"pos": np.zeros(3), "rot": np.eye(3), "size": 100, "focal": 80.0}
    uv, _ = project(cam, np.array([[0.5, 0.5, -2.0]]))
    assert uv[0, 0] > 50 and uv[0, 1] < 50


def test_points_behind_the_camera_or_behind_a_nearer_surface_are_not_visible():
    cam = {"pos": np.zeros(3), "rot": np.eye(3), "size": 10, "focal": 10.0}
    uv, depth = project(cam, np.array([[0.0, 0.0, -2.0], [0.0, 0.0, -3.0], [0.0, 0.0, 2.0]]))
    depth_map = np.full((10, 10), 2.0)
    assert visible(uv, depth, depth_map).tolist() == [True, False, False]


def test_camera_from_reads_the_pose_of_a_named_camera():
    class Model:
        cam_fovy = np.array([90.0])

        def camera_name2id(self, name):
            return 0

    class Data:
        cam_xpos = np.array([[1.0, 2.0, 3.0]])
        cam_xmat = np.eye(3).reshape(1, 9)

    class Sim:
        model, data = Model(), Data()

    class Env:
        sim = Sim()

    cam = camera_from(Env(), "any", 64)
    assert np.allclose(cam["pos"], [1, 2, 3]) and np.isclose(cam["focal"], 32.0)
