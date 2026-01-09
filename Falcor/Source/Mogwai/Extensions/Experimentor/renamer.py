import os

# Specify the directory containing the .png files
directory = 'D:/GradientDiffusion/gradientdiffusion/Falcor/Source/Mogwai/Extensions/Experimentor/TMP/'

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a PNG
    if filename.endswith('.png'):
        # Construct the full file path
        old_file = os.path.join(directory, filename)
        # Change the file extension to .bmp
        new_file = os.path.join(directory, filename[:-4] + '.bmp')
        # Rename the file
        os.rename(old_file, new_file)
        print(f'Renamed: {old_file} to {new_file}')

print('Renaming complete.')
