docker build --platform linux/arm64/v8 -t chalkroll:latest -f docker/Dockerfile .
docker save -o chalkroll_arm64.tar chalkroll:latest