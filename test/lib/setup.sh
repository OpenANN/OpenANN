if [ $# -ne 1 ]; then
  echo "usage: setup.sh TARGET_DIR" 1>&2
  exit 1
fi

TARGET_DIR=$1

echo "Starting CPP-Test setup script."
if [ -d "${TARGET_DIR}" ]; then
  echo "Removing current library."
  rm -r ${TARGET_DIR}
fi
echo "Downloading CPP-Test."
wget https://github.com/AlexanderFabisch/CPP-Test/zipball/master
echo "Unzipping library."
unzip master > /dev/null
rm master
mv *-CPP-Test-* ${TARGET_DIR}
echo "Successfully installed CPP-Test."
exit 0
