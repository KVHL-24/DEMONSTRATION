"""
main.py

Usage:
"""

import sys
from PySide6.QtWidgets import QApplication

from data_pipeline import SyntheticDataPipeline
from gui import GUI


def main():

    stream = SyntheticDataPipeline(
        "../../../data/directional_near_dynamic_taz+022_iaz-032_snr+0_rep09/array_audio.wav", 
        "../../../data/directional_near_dynamic_taz+022_iaz-032_snr+0_rep09/gaze.npy",
        "../../../data/directional_near_dynamic_taz+022_iaz-032_snr+0_rep09/metadata.json"
    )

    app = QApplication(sys.argv)

    player = GUI(stream)
    player.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()