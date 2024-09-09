sudo raspi-config > enable serial

/boot/config.txt

    enable_uart=1
    dtoverlay=pi3-miniuart-bt

both steppers are driven by the same driver but one needs to rotate in the other direction to drive the robot forwards. Use tweezers to remove the crimped cables from the 6-pin stepper motor connector for the upper X driver and flip them.