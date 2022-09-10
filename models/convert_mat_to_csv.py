import numpy as np
from scipy.io import loadmat
from pandas import DataFrame


class Read_Paderborn_Mat():
    """Read mat file from paderborn dataset
    """

    def __init__(self, file_path: str) -> None:
        """Init the class with file path.
            Read 64kHz data: phase_current_1, phase_current_2, vibration.
            Read 4kHz data: force, torque, speed
            Read 1kHz data: temp

        Args:
            file_path (str): source file path of .mat from Paderborn dataset
        """
        mat_file = loadmat(file_path, simplify_cells=True)
        mat = {k: v for k, v in mat_file.items() if k[0] != "_"}
        mat = mat[file_path[file_path.rfind("\\") + 1:-4]]["Y"]
        name = [row["Name"] for row in mat]
        data = [row["Data"] for row in mat]
        feature_dict = dict()
        for n, d in zip(name, data):
            feature_dict[n] = d

        # 4kHz
        self.force = feature_dict["force"]
        self.speed = feature_dict["speed"]
        self.torque = feature_dict["torque"]
        # 64kHz
        self.phase_current_1 = feature_dict["phase_current_1"]
        self.phase_current_2 = feature_dict["phase_current_2"]
        self.vibration = feature_dict["vibration_1"]
        # 1kHz
        self.temp = feature_dict["temp_2_bearing_module"]
        self.normalize_data()

    def normalize_data(self):
        """Normalize data into the integer second in each sampling rate
        """
        measure_time_floor = np.floor([len(self.vibration) / 64000])[0]
        length_floor_64kHz = int(measure_time_floor * 64000)
        length_floor_4kHz = int(measure_time_floor * 4000)
        length_floor_1kHz = int(measure_time_floor)

        # normalize to integer sec
        self.force = self.force[:length_floor_4kHz]
        self.speed = self.speed[:length_floor_4kHz]
        self.torque = self.torque[:length_floor_4kHz]

        self.phase_current_1 = self.phase_current_1[:length_floor_64kHz]
        self.phase_current_2 = self.phase_current_2[:length_floor_64kHz]
        self.vibration = self.vibration[:length_floor_64kHz]

        self.temp = self.temp[:length_floor_1kHz]

    def to_df_up_sample(self) -> DataFrame:
        """Up sampling dataset into 64kHz

        Returns:
            DataFrame: seven labels table (eight labels if include index) and each of label are up-sampled
        """
        # up-sample
        force = np.repeat(self.force, 16)
        speed = np.repeat(self.speed, 16)
        torque = np.repeat(self.torque, 16)

        temp = np.repeat(self.temp, 64000)

        return DataFrame({"vibration": self.vibration, "phase_current_1": self.phase_current_1,
                          "phase_current_2": self.phase_current_2, "force": force, "speed": speed,
                          "torque": torque, "temp": temp})

    def to_df_down_sample(self) -> DataFrame:
        """Down sampling dataset into 4kHz (temp is up-sample as it is 1kHz)

        Returns:
            DataFrame: seven labels table(eight labels if include index) and each of label are down-sampled
        """
        # down-sample
        vibration = self.vibration[::16]
        phase_current_1 = self.phase_current_1[::16]
        phase_current_2 = self.phase_current_2[::16]

        # up-sample temp
        temp = np.repeat(self.temp, 4000)

        return DataFrame({"vibration": vibration, "phase_current_1": phase_current_1,
                          "phase_current_2": phase_current_2, "force": self.force,
                          "speed": self.speed, "torque": self.torque, "temp": temp})
