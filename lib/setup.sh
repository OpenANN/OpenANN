if [ $# -ne 2 ]; then
  echo "usage: setup.sh TARGET_CMAES TARGET_ALGLIB" 1>&2
  exit 1
fi

TARGET_CMAES=$1
TARGET_ALGLIB=$2

echo "Starting setup script for external libraries."
if [ -d "${TARGET_CMAES}" ]; then
  echo "Removing current CMA-ESpp library."
  rm -r ${TARGET_CMAES}
fi
echo "Downloading CMA-ESpp..."
if wget --no-check-certificate https://github.com/AlexanderFabisch/CMA-ESpp/zipball/master; then
  echo "Success."
else
  echo "Failed."
  exit 1
fi
echo "Unzipping CMA-ESpp."
if unzip master > /dev/null; then
  echo "Success."
else
  echo "Failed."
  exit 1
fi
rm master
mv *-CMA-ESpp-* ${TARGET_CMAES}
echo "Successfully installed CMA-ESpp."

if [ -d "${TARGET_ALGLIB}" ]; then
  echo "Removing current ALGLIB library."
  rm -r ${TARGET_ALGLIB}
fi
echo "Downloading ALGLIB."
if wget https://www.alglib.net/translator/re/alglib-3.5.0.cpp.zip; then
  echo "Success."
else
  echo "Failed."
  exit 2
fi
echo "Unzipping ALGLIB."
unzip alglib-3.5.0.cpp.zip > /dev/null
rm alglib-3.5.0.cpp.zip
mv cpp/src ${TARGET_ALGLIB}
mv cpp/gpl* ${TARGET_ALGLIB}
rm -r cpp
echo "Successfully installed ALGLIB."
exit 0
