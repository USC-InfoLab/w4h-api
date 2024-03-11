import openai
import datetime
import yaml
import sqlalchemy
import urllib.parse
import os
import datetime
from datetime import datetime as dt
from datetime import timedelta
from datetime import time as dt_time
import sqlite3
import json
import pickle
import streamlit as st
from loguru import logger
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, ForeignKey, DateTime, REAL, Integer, Float, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, create_database
from geoalchemy2 import Geometry

import plotly.express as px
import plotly.graph_objs as go



# -------------- nav.py --------------
def createNav():
    st.sidebar.title("W4H Integrated Toolkit")
    # Using object notation
    isLogin = st.session_state.get("login-state",False)
    loginPage = st.sidebar.button('Log Out' if isLogin else 'Log In',use_container_width=True,type="primary")

    st.sidebar.divider()
    st.sidebar.caption("Import Historical Data / Instantiate a New W4H DB Instance")
    importPage = st.sidebar.button("ImportHub",use_container_width=True,type="secondary")

    st.sidebar.divider()
    st.sidebar.caption("Dashboard / Analyze Subjects Data")

    inputPage = st.sidebar.button("Input Page",use_container_width=True,type="secondary")
    resultPage = st.sidebar.button("Result Page",use_container_width=True,type="secondary")
    queryHistory = st.sidebar.button("Query History",use_container_width=True,type="secondary")

    st.sidebar.divider()
    st.sidebar.caption("Tutorial")
    tutorial = st.sidebar.button("How to Start",use_container_width=True,type="secondary")

    if (loginPage):
        if(isLogin):
            st.session_state["login-state"] = False
        st.session_state["page"] = "login"
        st.experimental_rerun()
    if (importPage):
        st.session_state["page"] = "import"
        st.experimental_rerun()
    if(inputPage):
        st.session_state["page"] = "input"
        st.experimental_rerun()
    if(resultPage):
        st.session_state["page"] = "result"
        st.experimental_rerun()
    if(queryHistory):
        st.session_state["page"] = "query_history"
        st.experimental_rerun()

    if(tutorial):
        st.session_state["page"] = "tutorial"
        st.experimental_rerun()

# -------------- query_history.py --------------
class query_history:
    def __init__(self,session):
        self.data = {}
        key_list = list(session.keys())
        for key in key_list:
            self.data[key] = session.get(key)

    def set(self, key,value):
        self.data[key] = value

    def get(self,key):
        return self.data[key]

    def setSession(self,session):
        key_list = list(self.data.keys())
        for key in key_list:
            session[key] = self.data.get(key)
        return session


# -------------- utils.py --------------
class Singleton(type):
    """Metaclass implementing the Singleton pattern.

    This metaclass ensures that only one instance of a class is created and shared among all instances.

    Attributes:
        _instances (dict): Dictionary holding the unique instances of each class.

    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Overrides the call behavior when creating an instance.

        This method checks if an instance of the class already exists. If not, it creates a new instance and
        stores it in the _instances dictionary.

        Args:
            cls (type): Class type.

        Returns:
            object: The instance of the class.

        """
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

def load_config(config_file: str) -> dict:
    """Read the YAML config file

    Args:
        config_file (str): YAML configuration file path
    """
    with open(config_file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def save_config(config_file, config):
    with open(config_file, 'w') as file:
        yaml.dump(config, file)

def getServerIdByNickname(config_file: str = 'conf/db_config.yaml', nickname='local db'):
    config = load_config(config_file)
    server_number = config['database_number']
    for i in range(1, server_number + 1):
        if (config["database" + str(i)]['nickname'] == nickname):
            return i
    raise Exception("No such nickname: \"" + nickname + "\"")

def get_db_engine(config_file: str = 'conf/db_config.yaml', db_server_id=1, db_server_nickname=None, db_name=None,
                  mixed_db_name=None) -> sqlalchemy.engine.base.Engine:
    """Create a SQLAlchemy Engine instance based on the config file

    Args:
        config_file (str): Path to the config file
        db_name (str, optional): Name of the database to connect to. Defaults to None.

    Returns:
        sqlalchemy.engine.base.Engine: SQLAlchemy Engine instance for the database

    """
    # load the configurations
    config = load_config(config_file=config_file)
    # Database connection configuration
    if mixed_db_name != None:
        db_server_nickname = mixed_db_name.split("] ")[0][1:]
        db_name = mixed_db_name.split("] ")[1]
        print(mixed_db_name, "!")
        print("server: ", db_server_nickname, "!")
        print("db_name: ", db_name, "!")
    if db_server_nickname != None:
        db_server_id = getServerIdByNickname(nickname=db_server_nickname)
    db_server = 'database' + str(db_server_id)
    dbms = config[db_server]['dbms']
    db_host = config[db_server]['host']
    db_port = config[db_server]['port']
    db_user = config[db_server]['user']
    db_pass = config[db_server]['password']
    db_name = db_name if db_name else ''

    db_user_encoded = urllib.parse.quote_plus(db_user)
    db_pass_encoded = urllib.parse.quote_plus(db_pass)

    # creating SQLAlchemy Engine instance
    con_str = f'postgresql://{db_user_encoded}:{db_pass_encoded}@{db_host}:{db_port}/{db_name}'
    db_engine = create_engine(con_str, echo=True)

    return db_engine

def parse_query(query, default_values):
    openai.api_key = os.environ['OPENAI_API_KEY']  # Replace with your OpenAI API key

    prompt = f"""
    Parse the user's query and update the structured data for a data analysis application. Pay special attention to the age range, weight range, and height range specified by the user for both the subjects (the ones user wants to show) and the control group (the ones user wants to compare with). Extract these specific ranges from the query and apply them to the control group parameters in the JSON object. Retain the default values for any variables not explicitly mentioned in the user's query.
    default values:

    For Subjects:
    - selected_users: Default is {default_values['selected_users']}
    - selected_state_of_residence: Default is {default_values['selected_state_of_residence']}
    - selected_age_range: Default is {default_values['selected_age_range']}
    - selected_weight_range: Default is {default_values['selected_weight_range']}
    - selected_height_range: Default is {default_values['selected_height_range']}

    For Control Group:
    - selected_users_control: Default is {default_values['selected_users_control']}
    - selected_state_of_residence_control: Default is {default_values['selected_state_of_residence_control']}
    - selected_age_range_control: Default is {default_values['selected_age_range_control']}
    - selected_weight_range_control: Default is {default_values['selected_weight_range_control']}
    - selected_height_range_control: Default is {default_values['selected_height_range_control']}

    For Analysis Time Frame:
    - start_date: Default is {default_values['start_date']}
    - end_date: Default is {default_values['end_date']}

    User Query: "{query}"

    Based on the query, subject attributes are the ones user wants to show. Control group attributes are the ones the user wants to compare with. You should be able to find the wanted age, weight, and height ranges based on the input. provide the values for all previous variables in a JSON object. If a variable is not specified in the query, retain its default value. Please don't output any other text than the json object.
    The returned json object must have only the following keys: {default_values.keys()}
    """
    # prompt = f"""
    # Parse the following user query into a JSON object representing structured data for a data analysis application. The application has these input variables for subjects and control groups, along with analysis time frame and specific time ranges, with their default values:

    # For Subjects:
    # - selected_users: is list of subjects user wants to show.
    # - selected_state_of_residence: is a list of states of residence user wants to show. values should be from the list of states in the dataset: {default_values['selected_state_of_residence']}
    # - selected_age_range: is a list with two elements, the first one is the lower bound of the age range, and the second one is the upper bound.
    # - selected_weight_range: is a list with two elements, the first one is the lower bound of the weight range, and the second one is the upper bound.
    # - selected_height_range: is a list with two elements, the first one is the lower bound of the height range, and the second one is the upper bound.

    # For Control Group:
    # This is the attributes of the control group that the user wants to compare with the subject group.
    # - selected_users_control: is list of subjects user wants to show.
    # - selected_state_of_residence_control: is a list of states of residence user wants to show. values should be from the list of states in the dataset: {default_values['selected_state_of_residence_control']}
    # - selected_age_range_control: is a list with two elements, the first one is the lower bound of the age range, and the second one is the upper bound.
    # - selected_weight_range_control: is a list with two elements, the first one is the lower bound of the weight range, and the second one is the upper bound.
    # - selected_height_range_control: is a list with two elements, the first one is the lower bound of the height range, and the second one is the upper bound.

    # For Analysis Time Frame:
    # - start_date: Default is {default_values['start_date']}
    # - end_date: Default is {default_values['end_date']}

    # User Query: "{query}"

    # Based on the query, subject attributes are the ones user wants to show. If a variable is not specified in the query, return None for the variable.
    # The returned json object must have only the following keys: {default_values.keys()}
    # """

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error in querying OpenAI: {e}")
        return None

# -------------- w4h_db_utils.py --------------
def create_tables(db_server_nickname: str, db_name: str, config_file='conf/config.yaml'):
    """Create the W4H tables in the database with the given name based on the config file

    Args:
        db_name (str): Name of the database to create the tables in
        config_file (str, optional): Path to the config file. Defaults to 'conf/config.yaml'.
    """
    metadata = MetaData()
    config = load_config(config_file=config_file)
    db_engine = get_db_engine(db_server_nickname=db_server_nickname, db_name=db_name)
    # try:
    columns_config = config["mapping"]["columns"]

    # Create the user table
    user_table_config = config["mapping"]["tables"]["user_table"]
    dtype_mappings = config['mapping']['data_type_mappings']
    user_columns = [eval(
        f'Column("{col_attribute["name"]}", {dtype_mappings[col_attribute["type"]]}, primary_key={col_attribute["name"] == columns_config["user_id"]})')
                    for col_attribute in user_table_config["attributes"]]  # Convert string to actual SQLAlchemy type
    user_table = Table(user_table_config["name"], metadata, *user_columns)

    # Create time series tables
    for table_name in config["mapping"]["tables"]["time_series"]:
        table = Table(table_name, metadata,
                      Column(columns_config["user_id"],
                             ForeignKey(user_table_config["name"] + '.' + columns_config["user_id"]), primary_key=True),
                      Column(columns_config["timestamp"], DateTime, primary_key=True),
                      Column(columns_config["value"], REAL),
                      )

    # Create geo tables
    for table_name in config["mapping"]["tables"]["geo"]:
        table = Table(table_name, metadata,
                      Column(columns_config["user_id"],
                             ForeignKey(user_table_config["name"] + '.' + columns_config["user_id"]), primary_key=True),
                      Column(columns_config["timestamp"], DateTime, primary_key=True),
                      Column(columns_config["value"], Geometry('POINT'))
                      )

    metadata.create_all(db_engine)
    # except Exception as err:
    #     db_engine.dispose()
    #     logger.error(err)


def create_w4h_instance(db_server: str, db_name: str, config_file='conf/config.yaml'):
    """Create a new W4H database instance with the given name and initialize the tables based on the config file

    Args:
        db_name (str): Name of the database to create
        config_file (str, optional): Path to the config file. Defaults to 'conf/config.yaml'.
    """
    db_engine_tmp = get_db_engine(db_server_nickname=db_server)
    try:
        logger.info('Database engine created!')
        # Execute the SQL command to create the database if it doesn't exist
        if not database_exists(f'{db_engine_tmp.url}{db_name}'):
            create_database(f'{db_engine_tmp.url}{db_name}')
            logger.success(f"Database {db_name} created!")
            db_engine_tmp.dispose()
        else:
            logger.error(f"Database {db_name} already exists!")
            db_engine_tmp.dispose()
            return
    except Exception as err:
        logger.error(err)
        db_engine_tmp.dispose()
    db_engine = get_db_engine(db_server_nickname=db_server, db_name=db_name)
    try:
        # Enable PostGIS extension
        with db_engine.connect() as connection:
            connection.execute(text(f"CREATE EXTENSION postgis;"))
            logger.success(f"PostGIS extension enabled for {db_name}!")
        db_engine.dispose()
    except Exception as err:
        logger.error(err)
        db_engine.dispose()
        return
    # Create the W4H tables
    create_tables(config_file=config_file, db_name=db_name, db_server_nickname=db_server)
    logger.success(f"W4H tables initialized!")


def get_existing_databases(config_file='conf/db_config.yaml') -> list:
    """Get a list of all existing databases

    Args:
        config_file (str, optional): Path to the config file. Defaults to 'conf/config.yaml'.

    Returns:
        list: List of all existing databases (strings)
    """
    db_list = []
    config = load_config(config_file=config_file)
    database_number = config['database_number']
    for i in range(1, database_number + 1):
        db_engine = get_db_engine(db_server_id=i)
        try:
            with db_engine.connect() as connection:
                result = connection.execute(text("SELECT datname FROM pg_database WHERE datistemplate = false;"))
                db_list += ['[' + config['database' + str(i)]['nickname'] + '] ' + row[0] for row in result]
            db_engine.dispose()
        except Exception as err:
            logger.error(err)
            db_engine.dispose()
            return db_list
    return db_list


def get_existing_database_server(config_file='conf/db_config.yaml') -> list:
    db_list_server = []
    config = load_config(config_file=config_file)
    database_number = config['database_number']
    for i in range(1, database_number + 1):
        db_list_server += [config['database' + str(i)]['nickname'] + ' (' + config['database' + str(i)]['host'] + ')']
    return db_list_server


def populate_tables(df: pd.DataFrame, db_name: str, mappings: dict, config_path='conf/config.yaml'):
    """Populate the W4H tables in the given database with the data from the given dataframe based on
    the mappings between the CSV columns and the database tables.

    Args:
        df (pd.DataFrame): Dataframe containing the data to be inserted into the database
        db_name (str): Name of the database to insert the data into
        mappings (dict): Dictionary containing the mappings between the CSV columns and the database tables
        config_path (str, optional): Path to the config file. Defaults to 'conf/config.yaml'.
    """
    # Load the config
    config = load_config(config_path)

    # Extract default column names from the config
    default_user_id = config['mapping']['columns']['user_id']
    default_timestamp = config['mapping']['columns']['timestamp']
    default_value = config['mapping']['columns']['value']
    user_table_name = config['mapping']['tables']['user_table']['name']

    # Create a session
    engine = get_db_engine(mixed_db_name=db_name)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Ensure all unique users from the dataframe exist in the user table
    unique_users = df[mappings[default_user_id]].unique().astype(str)
    existing_users = session.query(
        Table(user_table_name, MetaData(bind=engine), autoload=True).c[default_user_id]).all()
    existing_users = [x[0] for x in existing_users]

    # Identify users that are not yet in the database
    new_users = set(unique_users) - set(existing_users)

    if new_users:
        # Convert the set of new users into a DataFrame
        all_new_users = pd.DataFrame({default_user_id: list(new_users)})

        # Use to_sql to insert all new users into the user table
        all_new_users.to_sql(user_table_name, engine, if_exists='append', index=False)

    # Get the subset of mappings that doesn't include default_user_id and default_timestamp
    table_mappings = {k: v for k, v in mappings.items() if k not in [default_user_id, default_timestamp]}

    # Loop through each table in table_mappings
    for table_name, csv_column in table_mappings.items():
        # Check if the mapping is not NULL and exists in the df
        if csv_column and csv_column in df.columns:

            # Ensure that the dataframe columns match the user_id, timestamp, and value from your CSV
            columns_to_insert = [mappings[default_user_id], mappings[default_timestamp], csv_column]

            subset_df = df[columns_to_insert].copy()

            # Rename columns to match the table's column names using the defaults from config
            subset_df.columns = [default_user_id, default_timestamp, default_value]

            # dropping duplicate user_id and timestamp
            subset_df.drop_duplicates(subset=[default_user_id, default_timestamp], inplace=True)
            # subset_df = subset_df.groupby([default_user_id, default_timestamp]).mean().reset_index()

            # handling geometry data
            if table_name in config["mapping"]["tables"]["geo"]:
                subset_df[default_value] = subset_df[default_value].apply(lambda x: f'POINT{x}'.replace(',', ''))

            # Insert data into the table
            subset_df.to_sql(table_name, engine, if_exists='append', index=False)

    # Commit the remaining changes and close the session
    session.commit()
    session.close()
    engine.dispose()


def populate_subject_table(df: pd.DataFrame, db_name: str, mappings: dict, config_path='conf/config.yaml'):
    """Populate the W4H tables in the given database with the data from the given dataframe based on
    the mappings between the CSV columns and the database tables.

    Args:
        df (pd.DataFrame): Dataframe containing the data to be inserted into the database
        db_name (str): Name of the database to insert the data into
        mappings (dict): Dictionary containing the mappings between the CSV columns and the database tables
        config_path (str, optional): Path to the config file. Defaults to 'conf/config.yaml'.
    """
    # Load the config
    config = load_config(config_path)

    # Create a session
    engine = get_db_engine(mixed_db_name=db_name)

    # create a user table dataframe using the mappings
    user_tbl_name = config['mapping']['tables']['user_table']['name']
    user_df = pd.DataFrame()
    for k, v in mappings.items():
        if v is not None:
            user_df[k] = df[v]
    # populate the user table (directly push df to table), if already exists, append new users
    # if columns don't exist, ignore
    user_df.to_sql(user_tbl_name, engine, if_exists='append', index=False)

    # Commit the remaining changes and close the session
    engine.dispose()


def getCurrentDbByUsername(username):
    with sqlite3.connect('user.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''select current_db from users where username = ?''', (username,))
        result = cursor.fetchone()
    return result[0]


def updateCurrentDbByUsername(username, currentDb):
    with sqlite3.connect('user.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''update users set current_db = ? where username = ?''', (currentDb, username,))
        conn.commit()


def saveSessionByUsername(session):
    with sqlite3.connect('user.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''select query_history from users where username = ?''', (session.data.get('login-username'),))
        result = cursor.fetchone()
        conn.commit()
    query_history = pickle.loads(result[0])
    # print("history:",query_history[0].get('selected_users'))
    query_history.append(session)
    serialized_object = pickle.dumps(query_history)

    with sqlite3.connect('user.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''UPDATE users SET query_history = ? WHERE username = ?''',
                       (serialized_object, session.data['login-username'],))
        conn.commit()


def getSessionByUsername(username):
    with sqlite3.connect('user.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''select query_history from users where username = ?''', (username,))
        result = cursor.fetchone()
        conn.commit()
    return pickle.loads(result[0])


# -------------- viz.py --------------
def get_bar_fig(df, label='Feature'):
    fig = px.bar(
                x=df.columns.tolist(),
                y=df.values.flatten().tolist()
    )

    fig.update_layout(
        width=250,
        height=300,
        showlegend=False,
        xaxis_title=None,
        yaxis_title=label,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    fig.update_traces(marker_color=['#636EFA', '#00B050'])

    return fig

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

def get_map_legend(color_lookup):
    # map_legend_lookup = [{'text': t, 'color': rgb_to_hex(c)} for t, c in color_lookup.items()]
    # legend_markdown = "<br>".join([f"<span style='color:{leg['color']}'> &#9679; </span>{leg['text']}" for leg in map_legend_lookup])
    # return st.markdown(legend_markdown, unsafe_allow_html=True)
    map_legend_lookup = [{'text': t, 'color': rgb_to_hex(c)} for t, c in color_lookup.items()]
    legend_markdown = "  \n".join([f"<span style='color:{leg['color']}'> &#9679; </span>{leg['text']}" for leg in map_legend_lookup])
    return st.markdown(f"<p style='font-size: 16px; font-weight: bold;'>Map Legend</p>{legend_markdown}", unsafe_allow_html=True)
