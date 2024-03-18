# Smart Renting: estimate the rental value 

Machine Learning project to predict rental values in São Carlos based on rented property characteristics, which include: property type (house/apartment), number of bedrooms, bathrooms, garages, suites, furnished (yes/no).

## Project structure:
<pre>
<code>
├── Aluguel_SaoCarlos_Python_Flask
│   ├── ml_model
│   │    ├── model.pkl
│   │    ├── tabela_aluguel_completa.csv
│   │    └── training.py
│   │        
│   ├── static  
│   │    ├── assets
│   │    │    ├── css
│   │    │    ├── js
│   │    │    ├── sass
│   │    │    └── webfonts
│   │    ├── images
│   │    │    ├── fulls
│   │    │    ├── thumbs
│   │    │    ├── avatar.jpg
│   │    │    └── bg.jpg
│   │    
│   ├── templates
│   │    ├── index.html
│   │    ├── LICENSE.txt
│   │    └── README.txt 
│   │        
│   ├── webscraping 
│   │    └── webscraping.ipynb        
│   │ 
├── app.py
├── README.md
└── requirements.txt
</code>
</pre>

- In <a href="https://github.com/naomyduarteg/Aluguel_SaoCarlos_Python_Flask/tree/main/ml_model">ml_model</a>, you will find the Python file used to prepare the data and train the ML model, which generates the file model.pkl, which will be used to predict values.
- In <a href="https://github.com/naomyduarteg/Aluguel_SaoCarlos_Python_Flask/tree/main/static">static</a>, you will find the files that generate the appearance of the website.
- In <a href="https://github.com/naomyduarteg/Aluguel_SaoCarlos_Python_Flask/tree/main/templates">templates</a>, you will find the file that defines the main structure of the website.
- In <a href="https://github.com/naomyduarteg/Aluguel_SaoCarlos_Python_Flask/tree/main/webscraping">webscraping</a>, you will find the Jupiter notebook file where the rental data scraping is performed.
- In <a href="https://github.com/naomyduarteg/Aluguel_SaoCarlos_Python_Flask/blob/main/app.py">app</a>, you will find the main file of the flask API that renders the template and executes the prediction using the selected ML model and the data entered by the user.

## Running the website 
1. Clone the repository

```
git@github.com:naomyduarteg/Aluguel_SaoCarlos_Python_Flask.git
```
2. Create a virtual environment

```
python3 -m venv <name_of_venv>
```
3. Go to the virtual environment's directory and activate it

For Windows:
```
Scripts/activate
```
For Linux/Mac:
```
bin/activate
```
4. Install the requirements

```
pip install -r requirements.txt
```
6. Run the Flask API
```
python flask_app.py
```

