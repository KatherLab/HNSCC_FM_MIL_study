source this/is/where/your/venv/is #Location of your python .venv for the invocation of TotalSegmentator

cd this/is/where/your/resampled/NIFTIscans/are/theyshouldnamed/image.nii #The script assumes that you are using images in nifti called image.nii
#The structure for the loop is:
#-Patient
    #--image.nii.gz
#This will output the body and extremities in the same subdirectory 
for dir in *
do
	echo "Processing directory: $dir"
	dir_red=$(basename "$dir")
	echo "Segmenting the body"
	TotalSegmentator -i "$dir/image.nii.gz" -o "$dir" --task body  --device gpu

done
