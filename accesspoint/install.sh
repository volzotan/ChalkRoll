# Access Point

sudo apt-get update && sudo apt-get install -y dnsmasq hostapd

sudo cp /home/pi/accesspoint/hostapd.conf /etc/hostapd/hostapd.conf
sudo cp /home/pi/accesspoint/dnsmasq.conf /etc/.

sudo systemctl unmask hostapd
sudo systemctl disable hostapd
sudo systemctl disable dnsmasq

sudo ln -s /home/pi/accesspoint/accesspoint.service /etc/systemd/system/accesspoint.service
sudo systemctl enable accesspoint.service
sudo systemctl start accesspoint.service
