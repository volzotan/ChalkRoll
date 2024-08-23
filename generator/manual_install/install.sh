pip3 install -r requirements.txt

sudo ln -s /home/pi/generator/gunicorn.service /etc/systemd/system/gunicorn.service
sudo systemctl enable gunicorn.service