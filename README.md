## Install Dependencies
```
apt-get install $(grep -vE "^\s*#" requirements.txt  | tr "\n" " ")
```

## Compile
```
cmake .
```
## Run the code to extract frames containing car movements
```
./CarExistenceCheck/CarLicensePlateDetection -d /media/???/DRKHORSANDI/Camera/Wednesday/
```
