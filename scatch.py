import deeplabcut
a
deeplabcut.analyze_videos('/data/common/dlc_models/eye/config.yaml', '/home/adamranson/dlc/test1/2023-05-15_06_ESMT134_eye1_left.avi', videotype='avi', shuffle=1, gputouse=0, save_as_csv=True, destfolder='/home/adamranson/dlc/test1/')