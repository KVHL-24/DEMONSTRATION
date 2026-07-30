import numpy as np
from scipy.io import wavfile
import math
import cv2
import json

MIC_PATTERNS = [
    [2], # 1 mic,
    [0, 5], # 2 mics,
    [0, 2, 5], # 3 mics,
    [0, 1, 4, 5], # 4 mics,
    [0, 1, 2, 4, 5], # 5 mics,
    [0, 1, 2, 3, 4, 5] # 6 mics
]

MIC_POSITIONS = np.array([
    [-0.030,  0.000,  0.045],
    [ 0.030,  0.000,  0.045],
    [ 0.030,  0.000, -0.030],
    [-0.030,  0.000, -0.030],
    [-0.075, -0.010, -0.010],
    [ 0.075, -0.010, -0.010],
], dtype=np.float64)

class SyntheticDataPipeline:
    def __init__(self, wav_file, gaze_file, metadata):
        self.audio_sample_rate, self.audio_data = wavfile.read(wav_file)
        self.audio_window = 2048*8
        self.num_audio_windows = math.ceil(self.audio_data.shape[0] / self.audio_window)
        self.audio_channels = self.audio_data.shape[1] if self.audio_data.ndim > 1 else 1
        self.audio_duration = self.audio_data.shape[0] / self.audio_sample_rate

        self.audio_dt = self.audio_window / self.audio_sample_rate
        
        self.camera_dt = 1/10
        
        self.gaze_data = None
        
        self.params = ["num_mics", "audio_amp", "soft_clip", "camera_fov", "denoising_type"]
        self.num_mics = 6
        
        self.audio_amp = 0
        self.soft_clip = 1
        
        self.camera_fov = 110
        self.denoising_type = 0 # Avg, Directional
        self.override_gaze = None
        self.camera_width, self.camera_height = 512, 512
        
        self.gaze_data = np.load(gaze_file)
        num_gazes = self.gaze_data.shape[0]
        self.gaze_dt = self.audio_duration / num_gazes
        
        with open(metadata, 'r') as metadata_file:
            metadata_json = json.load(metadata_file)
            
        self.interferer_pose = [
            metadata_json["interferer_azimuth_deg"],
            0,
            0.5
        ]
        self.target_pose = [
            metadata_json["target_azimuth_deg"],
            0,
            0.5
        ]
        
        
    
    def get_param_bounds(self):
        return {
            "num_mics": (1, 6, 1),
            "audio_amp": (0, 50, 0.1),
            "soft_clip": (0, 1, 1),
            "camera_fov": (30, 180, 1),
            "denoising_type": (0, 2, 1)
        }

    
    def get_param(self, param):
        assert param in self.params
        return getattr(self, param)
    
    
    def set_param(self, param, value):
        assert param in self.params
        min_val, max_val, _ = self.get_param_bounds()[param]
        assert min_val <= value <= max_val
        setattr(self, param, value)


    def get_duration(self):
        return self.audio_duration
    
    
    def _proc_audio(self, x: np.ndarray, gaze: np.ndarray) -> np.ndarray:
        #left_channel = x[:, :3].mean(axis=1)
        #right_channel = x[:, 3:6].mean(axis=1)
        #two_channel = np.stack((left_channel, right_channel), axis=1)
        
        mics_used = MIC_PATTERNS[int(self.num_mics) - 1]
        x = x[:, mics_used]
        
        amp = 10**(self.audio_amp/20)
        INT32_MAX = np.iinfo(np.int32).max
        if self.soft_clip > 0:
            x_double = x.astype(np.float64) #/ INT32_MAX
            x_double = np.tanh(amp * x_double)
            x = (x_double * INT32_MAX)
        else:
            x = ((amp * x.clip(-1/amp, 1/amp)) * INT32_MAX)
            
        if self.denoising_type == 0:
            x = x.mean(axis=1, keepdims=True)
        elif self.denoising_type == 1:
            mic_positions = MIC_POSITIONS[mics_used]
            x = self.delay_and_sum(x, gaze[0], gaze[1], self.audio_sample_rate, mic_positions)
            x = x.reshape(-1, 1)
        elif self.denoising_type == 2:
            x = self.do_fancy_denoising_stuff(x, gaze)

        return x.astype(np.int32)


    def delay_and_sum(self, audio, azimuth, elevation, sample_rate, mic_positions):
        # this is a vibecoded chatgpt generated beamforming thing just as a dummy / test
        # don't know how well it works but sounds quite crackly, probably not great

        c = 343.0  # speed of sound

        # Unit vector pointing towards sound source
        direction = np.array([
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation)
        ])

        # Projection of mic positions onto wave direction
        # (relative propagation delays)
        delays = mic_positions @ direction / c

        # Convert to samples
        delay_samples = delays * sample_rate

        # Remove arbitrary offset so all delays are positive
        delay_samples -= delay_samples.min()

        N = len(audio)
        output = np.zeros(N)

        for i, d in enumerate(delay_samples):
            # fractional delay by interpolation
            indices = np.arange(N) - d

            output += np.interp(
                indices,
                np.arange(N),
                audio[:, i],
                left=0,
                right=0
            )

        output /= len(MIC_POSITIONS)

        return output


    def do_fancy_denoising_stuff(self, x, gaze):
        # TODO: karen's algorithm?
        return x


    def _proc_camera(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def fetch_audio(self, t):
        audio_count = self.num_audio_windows
        
        window_idx = max(0, min(int(t / self.audio_dt), max(0, audio_count - 1)))
        i0 = window_idx * self.audio_window
        i1 = min(i0 + self.audio_window, self.audio_data.shape[0])
        next_t = i1*self.audio_dt/self.audio_window
        audio_data = self.audio_data[i0:i1]
        
        gaze = self.fetch_gaze(t)
        return self._proc_audio(audio_data, gaze), next_t
    
    

    def _project_point(self, azimuth_deg, elevation_deg, hfov_deg, width, height):
        az = np.deg2rad(azimuth_deg)
        el = np.deg2rad(elevation_deg)
        hfov = np.deg2rad(hfov_deg)

        plane_width = 2 * np.tan(hfov / 2)
        plane_height = plane_width * height / width

        px = np.tan(az)
        py = np.tan(el) / np.cos(az)

        x = (px / plane_width + 0.5) * width - 0.5
        y = (0.5 - py / plane_height) * height - 0.5

        return int(round(x)), int(round(y))


    def _draw_object(self, img, azimuth, elevation, distance, color,
                    hfov=None, radius_at_1m=20):
        if hfov is None:
            hfov = self.camera_fov
            
        h, w = img.shape[:2]

        x, y = self._project_point(azimuth, elevation, hfov, w, h)

        radius = max(2, int(radius_at_1m / distance))

        if -radius < x < w + radius and -radius < y < h + radius:
            cv2.circle(img, (x, y), radius, color, -1)

    def _draw_microphones(
        self,
        img,
        active_mic_list,
        num_mics=6,
        square_size=28,
        bottom_margin=20,
    ):
        h, w = img.shape[:2]

        # Evenly spaced x positions
        xs = np.linspace(
            square_size // 2 + 20,
            w - square_size // 2 - 20,
            num_mics,
        )

        cy = h - bottom_margin - square_size // 2

        active_color = (0, 0, 255)      # blue (BGR)
        inactive_color = (192, 192, 192)

        active = set(active_mic_list)

        for i, cx in enumerate(xs):
            cx = int(round(cx))

            x1 = cx - square_size // 2
            y1 = cy - square_size // 2
            x2 = cx + square_size // 2
            y2 = cy + square_size // 2

            color = active_color if i in active else inactive_color

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=-1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)

        
    def fetch_camera(self, t):
        # dummy data
        camera_data = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        
        self._draw_object(camera_data, azimuth=self.interferer_pose[0], elevation=self.interferer_pose[1], distance=self.interferer_pose[2], color=(255, 0, 0))
        self._draw_object(camera_data, azimuth=self.target_pose[0], elevation=self.target_pose[1], distance=self.target_pose[2], color=(0, 255, 0))
        active_mic_list = MIC_PATTERNS[int(self.num_mics) - 1]
        self._draw_microphones(camera_data, active_mic_list, num_mics=6)
        
        next_t = t + self.camera_dt
        return self._proc_camera(camera_data), next_t
        

    def set_gaze_xy(self, x, y):
        width = self.camera_width
        height = self.camera_height

        hfov = math.radians(self.camera_fov)

        # Physical size of image plane at z=1
        plane_width = 2 * math.tan(hfov / 2)
        plane_height = plane_width * height / width

        # Pixel -> image plane coordinates
        px = ((x + 0.5) / width - 0.5) * plane_width
        py = -(((y + 0.5) / height - 0.5) * plane_height)  # +y up
        pz = 1.0

        # Spherical coordinates
        azimuth = math.degrees(math.atan2(px, pz))
        elevation = math.degrees(math.atan2(py, math.sqrt(px**2 + pz**2)))
        radius = math.sqrt(px**2 + py**2 + pz**2)

        self.override_gaze = (azimuth, elevation, radius)
        

    def fetch_gaze(self, t, return_projection=False, ignore_override=False):
        if self.override_gaze is not None and not ignore_override:
            azimuth, elevation, radius = self.override_gaze
        else:
            gaze_idx = min(int(t / self.gaze_dt), self.gaze_data.shape[0] - 1)
            azimuth, elevation, radius = self.gaze_data[gaze_idx]
            # convert to degrees
            azimuth = np.rad2deg(azimuth)
            elevation = np.rad2deg(elevation)
            
        
        if return_projection:
            x, y = self._project_point(azimuth, elevation, self.camera_fov, 512, 512)
            return [azimuth, elevation, radius], [x, y]
        else:
            return [azimuth, elevation, radius]
        
        