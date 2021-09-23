mkdir -p $1"/training_data/"
wget --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1rMXPEG-VwYcVOTw7WWT2RwFSEeWn7n-Y' -O $1"training_data.zip"
unzip -n -q $1"training_data.zip" -d $1"/training_data/"
rm $1"training_data.zip"