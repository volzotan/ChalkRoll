import logging
import subprocess
import os
import time

"""
This script should be invoked at startup to check if the Raspberry Pi is able to
conntect to a known wifi network. If this fails after `WAIT_TIME` seconds, the Pi
is in the field and no networks are available. In this case start the necessary
tools to create an ad-hoc network (dnsmasq as DHCP server and hostapd as the access
point provider)
"""

LOG_FILE = "/home/pi/accesspoint/accesspoint.log"
WAIT_TIME = 40

if __name__ == "__main__":

    # create logger
    log = logging.getLogger()
    log.handlers = [] # remove externally inserted handlers (systemd?)
    log.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter("%(asctime)s | %(name)-7s | %(levelname)-7s | %(message)s")

    # console handler and set level to debug
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(formatter)
    log.addHandler(consoleHandler)

    fileHandlerDebug = logging.FileHandler(LOG_FILE, mode="a", encoding="UTF-8")
    fileHandlerDebug.setLevel(logging.DEBUG)
    fileHandlerDebug.setFormatter(formatter)
    log.addHandler(fileHandlerDebug)

    log.info("------------------------")
    log.info("accesspoint service init")

    # log.debug(subprocess.run(["sudo", "systemctl", "status", "hostapd"], capture_output=True).stdout)

    # waiting a bit
    time.sleep(WAIT_TIME)

    if os.uname().sysname == "Linux":

        # check if a successful connection to a known wifi network has happened:
        try:
            log.info("check for existing connection to wifi network")
            res = subprocess.run(["iwgetid", "-r"], capture_output=True)
            network_name = res.stdout.decode("utf-8")

            # iwgetid returns nothing if no network is connected        
            if res.returncode != 0 or len(res.stdout) == 0:
                log.info("creating accesspoint...")

                subprocess.run(["wpa_cli", "terminate"])
                subprocess.run(["sudo", "systemctl", "start", "dnsmasq"])
                subprocess.run(["sudo", "systemctl", "start", "hostapd"])

                # alternatively invoke hostapd directly:
                # subprocess.run(["sudo", "systemctl", "stop", "wpa_supplicant"], check=True)
                # subprocess.run(["sudo", "ifconfig", "wlan0", "down"], check=True)
                # subprocess.run(["sudo", "hostapd", "-B", "/etc/hostapd/hostapd.conf"], check=True)

                log.info("accesspoint created")
            else:
                log.info(f"no accesspoint required. connected to: {network_name}")

        except Exception as e:
            log.error(f"creating accesspoint failed: {e}")
