from paygapp import app
import pandas as pd
import MySQLdb
from flask import redirect,request,url_for,render_template
from MySQLdb import escape_string as thwart
import gc
import pickle
import os
#from models import File
# from db_client import DB_Client
# from flask import render_template, url_for, request, redirect
# from werkzeug.utils import secure_filename
from analysis_pred_model.GPG_Analytics_PGanalysis_module import PG_analysis
from analysis_pred_model.GPG_Analytics_Pred_module import Salary_Prediction

db = MySQLdb.connect(host="localhost",
                           user = "admin",
                           passwd = "admin",
                           db = "paygapp")
cur = db.cursor()


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['csv'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/paygap',methods=['GET', 'POST'])
def paygap():
    input_employee_data = pd.read_pickle("paygapp/Pickle_uploaded/updated_data.pkl")
    with open('paygapp/Pickle_uploaded/threshold.pkl', 'rb') as f:
        threshold = pickle.load(f)
    Pay_gap_graph_names, Pay_gap_graph_values, Pay_gap_table_stats, Final_comment = PG_analysis(input_employee_data,
                                                                                                threshold)
    try:
        max_values = max(Pay_gap_graph_values)
    except:
        max_values = 3
    data = Pay_gap_table_stats.to_dict('records')
    columns = Pay_gap_table_stats.columns
    return render_template('chart1.html', values=Pay_gap_graph_values, labels=Pay_gap_graph_names,
                           Comments=Final_comment, maxValues=max_values, data=data, columns=columns)


@app.route('/before_predict_salary',methods=['GET','POST'])
def before_predict_salary():
    return render_template('form_predict_salary.html')

@app.route('/predict_salary',methods=['GET','POST'])
def predict_salary():
    if request.method =='GET':
        return render_template('form_predict_salary.html')
    else:
        edu = request.form.get('Edu')
        jobTitle = request.form.get('JobTitle')
        experience = request.form.get('Exp')
        perfEval = request.form.get('PerfEval')
        employee_data = pd.read_pickle("paygapp/Pickle_uploaded/updated_data.pkl")
        with open('paygapp/Pickle_uploaded/threshold.pkl', 'rb') as f:
            GPG_threshold = pickle.load(f)

        Salary = Salary_Prediction(employee_data, GPG_threshold, jobTitle, perfEval, experience, edu)
        print Salary
        labels = ["march","april","may"]
        values = ["2","1","3"]
        data = employee_data.to_dict('records')
        columns = employee_data.columns

        return render_template('dashboard_final.html',data=data,columns=columns)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('cards.html')

    if request.method == 'POST':
        # check if the post request has the file part
        threshold = request.form.get("threshold_text")
        with open('paygapp/Pickle_uploaded/threshold.pkl', 'wb') as f:
            pickle.dump(threshold, f)
        if 'file' not in request.files:
           # flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            #flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            #file = File(file_path)
            #file.create_table()
            #file.upload_to_db()
           # return redirect(url_for('uploaded_file', filename=filename.split('.')))
            return redirect(url_for('uploaded_file', filename=filename))

@app.route('/',methods=["POST","GET"])
def index():
    return render_template("index.html")

@app.route('/before_forgetpass',methods=["POST","GET"])
def before_forgetpass():
    return render_template("forgot-password.html")

@app.route('/register',  methods=["POST","GET"])
def register():

        if request.method == "POST":
            email = request.form.get('user_id')
            password= request.form.get('password')
            cur.execute("SELECT COUNT(1) FROM users WHERE email = %s;", [email])
            # CHECKS IF USERNAME EXSIST
            if cur.fetchone()[0]:
                return '''
                <!doctype html>
                <h1>This email is already register. Please go back to login</h1>
                '''
            query = "SELECT * FROM users WHERE email = '%s'" % thwart(email)
            result = cur.execute(query)
            if int(result) > 0:
                return '''
                <!doctype html>
                <h1>That username is already taken, please choose another</h1>
                '''
            else:
                cur.execute("INSERT INTO users (password, email) VALUES (%s, %s)",
                          (thwart(password), thwart(email)))
                db.commit()
                #db.close()
                gc.collect()

                # session['logged_in'] = True
                # session['email'] = email
                # print "session value: ",session

                #return "Thanks for registering!"
                return redirect(url_for("login"))
        else:
            return render_template("register.html")

@app.route('/before_login',methods=['POST','GET '])
def before_login():

    return render_template('login.html')


@app.route('/login', methods=['POST','GET'])
def login():
    if request.method == 'POST':

        email_form = request.form.get('username_login')
        password_form = request.form.get('password_login')

        cur.execute("SELECT COUNT(1) FROM users WHERE email = %s;", [email_form])
        # CHECKS IF USERNAME EXSIST
        c= cur.fetchone()[0]
        if (c):
            cur.execute("SELECT password FROM users WHERE email = %s;", [email_form]) # FETCH THE HASHED PASSWORD
            for row in cur.fetchall():
                if(password_form==row[0]):
                   # session['email'] = request.form['email']
                   # session['logged_in'] = True
                    #flash("hello")
                    return render_template('dashboard.html')
                else:
                    return '''
                        <!doctype html>
                        <h1>Invalid Credentials! Go back to login or register</h1>
                        '''
        else:
            return redirect(url_for("register"))
    else:
        return render_template("login.html")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    #query = "SELECT * FROM {table_name}".format(table_name=filename)
    #df = db.get_dataframe(query)
    data_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    df = pd.read_csv(data_file_path)
    data = df.to_dict('records')
    columns = df.columns
    df.to_pickle("paygapp/Pickle_uploaded/updated_data.pkl")

    return render_template("tables.html", data=data, columns=columns)


