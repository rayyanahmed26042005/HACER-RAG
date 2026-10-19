@echo off
if not defined CMAKE_VERSION set CMAKE_VERSION=3.31.6
echo 🔧 Stopping all Gradle daemons...
cd android
call gradlew --stop

echo 🧹 Cleaning project...
call gradlew clean

echo 🗑 Deleting Gradle and build caches...
cd ..
rmdir /s /q ".gradle"
rmdir /s /q "android\.gradle"
rmdir /s /q "android\app\build"
rmdir /s /q "android\build"
if exist "node_modules\react-native-reanimated\android\.cxx" rmdir /s /q "\\?\node_modules\react-native-reanimated\android\.cxx"
if exist "android\app\.cxx" rmdir /s /q "\\?\android\app\.cxx"

echo 🚀 Running the app...
npx react-native run-android

pause
