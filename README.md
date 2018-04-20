To run the flask server do following:
```
export FLASK_APP=paygapp
flask run
```

Install MySQL server
```
sudo apt-get install mysql-server
```

Open up mysql:
```
sudo mysql

It will bring you to this prompt
mysql>
```

Create a database:
```
mysql> CREATE DATABASE paygapp
```

Create a user and grant all access
```
mysql> CREATE USER 'admin'@'localhost' IDENTIFIED BY 'admin';
mysql> GRANT ALL PRIVILEGES ON *.* TO 'admin'@'localhost' WITH GRANT OPTION;
