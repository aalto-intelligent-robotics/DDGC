mkdir -p $1"/test_data/"
wget --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=168ZCxdLOkuG_RehhpLrma5CqYBLDrXZD' -O $1"test_data.zip"
unzip -n -q $1"test_data.zip" -d $1"/test_data/"
rm $1"test_data.zip"