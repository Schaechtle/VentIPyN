from plotting import get_dataFrame_median
date_experiment= "t04_median_recording"
ini_file_path = "test_median_recording.ini"

df  = get_dataFrame_median(date_experiment,ini_file_path)

print(df)