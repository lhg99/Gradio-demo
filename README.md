# **Gradio-Demo**

## - **Gradio**
[<img src="readme_files/gradio.svg" alt="gradio" width=300>](https://gradio.app)<br>
Here's the gradio github link: https://github.com/gradio-app/gradio

Gradio is open-source python library. It can make demo for machine learning model.

### Gradio is useful for:

- **Demoing** your machine learning models for clients/collaborators/users/students.

- **Deploying** your models quickly with automatic shareable links and getting feedback on model performance.

- **Debugging** your model interactively during development using built-in manipulation and interpretation tools.

We can use various components for making demos.

This project provides demo for people who want to use gradio library.

## - **Demo Description**

This demo shows 2022 year weather in table and graph by selecting the month, day, weather elements, and location.

### **Input components**

Month, day : dropbox<br>
weather elements : checkboxGroup<br>
location : radio<br>
precipitation : checkbox

Precipitation data has few valid data, so we made a separate component without drawing a graph.

### **Output components**

Weather table : dataframe<br>
Weather graph : plots

We used pandas and matplotlib python library when we make dataframe and plots.

Humidity is represented by a bar graph and other weather elements are represented by a curved line graphs.

### **Function**

In demo, we made two functions which names are dataSearch and showOutput.

- dataSearch()

This function provides obtaining data about the month, day, weather element, and location entered by the user.

The columns in the csv file are composed of location, date, time, percipitation, wind, humidity, air pressure, and temperature. We took  csv files from weather data opening portal.

```python
def dataSearch(month, day, weather_elements, location, precipitation):
    if location=='Seoul':
        df = pd.read_csv('Seoul.csv')
    elif location=='Washington':
        df = pd.read_csv('Washington.csv')

    if precipitation:
        weather_elements.append('precipitation')

    if day in ['1','2','3','4','5','6','7','8','9']:
        today = '2022-'+month +'-0'+day
    else:
        today = '2022-'+month+'-'+day

    df1 = df[df.date == today]
    columns = ['location', 'date', 'time'] + weather_elements
    df2 = df1.loc[:, columns]
  
    return df2
```

- showOutput()

This function represents data obtained from dataSearch() by dataframe and graphs.

It replaced month with numeric form and the maximum number of plots increased to 2 depending on the weather elements selected.

```python
def showOutput(month, day, weather_elements, location, precipitation):
    if month=='January':
        month = '01'
    elif month=='February':
        month = '02'
    elif month=='March':
        month = '03'
    elif month=='April':
        month = '04'
    elif month=='May':
        month = '05'
    elif month=='June':
        month = '06'
    elif month=='July':
        month = '07'
    elif month=='August':
        month = '08'
    elif month=='September':
        month = '09'
    elif month=='October':
        month = '10'
    elif month=='November':
        month = '11'
    elif month=='December':
        month = '12'

    weatherTable = dataSearch(month, day, weather_elements, location, precipitation)

    if precipitation:
        weather_elements.remove('precipitation')

    if day in ['1','2','3','4','5','6','7','8','9']:
        xname = '2022-'+month +'-0'+day
    else:
        xname = '2022-'+month+'-'+day

    y_value=[0]*len(weather_elements)

    for i in range(len(weather_elements)):
        y_value[i] = weatherTable[weather_elements[i]]

    x_value = weatherTable['time']

    if 'humidity' in weather_elements:
        humidity_index = weather_elements.index('humidity')
        if weather_elements[humidity_index] != weather_elements[-1]:
            temp = weather_elements[humidity_index]
            weather_elements[humidity_index] = weather_elements[-1]
            weather_elements[-1] = temp


        if len(weather_elements) == 1:
            weatherPlot = plt.figure(figsize=(10,10))
            plt.title("2022 Weather Graph", fontsize=20, fontweight='bold')
            plt.xlabel(xname,labelpad=5, fontsize=15)
            plt.ylabel(weather_elements[0], labelpad=15, fontsize=15)
            plt.xticks(size=10, rotation=45)

            plt.bar(x_value, y_value[-1], color='skyblue', label='1st data')
            plt.legend(loc = "upper left")

        elif len(weather_elements) == 2:
            weatherPlot, ax1 = plt.subplots(figsize=(10,10))
            plt.title("2022 Weather Graph", fontsize=20, fontweight='bold')
            plt.xticks(size=10, rotation=45)

            ax1.bar(x_value, y_value[-1], color='skyblue', label='1st data')
            ax1.set_xlabel(xname, labelpad=5, fontsize=15)
            ax1.set_ylabel(weather_elements[1], labelpad=15, fontsize=15)
            ax1.legend(loc='upper left')

            ax1_sub = ax1.twinx()
            ax1_sub.plot(x_value, y_value[0], color='red', marker = "o", label='2nd data')
            ax1_sub.set_ylabel(weather_elements[0], labelpad=25, fontsize=15, rotation=270)
            ax1_sub.legend(loc='upper right')
            
        elif len(weather_elements) == 3:
            weatherPlot, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,15), constrained_layout=True)

            line1 = ax1.plot(x_value, y_value[0], color='red', marker = "o", label='1st data')
            ax1.set_xlabel(xname, labelpad=5, fontsize=15)
            ax1.set_ylabel(weather_elements[0], labelpad=15, fontsize=15)
            ax1.set_title("2022 Weather Graph", fontsize=20, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45, labelsize=10)

            ax1_sub = ax1.twinx()
            line2 = ax1_sub.plot(x_value, y_value[1], color='blue', marker = "o", label='2nd data')
            ax1_sub.set_ylabel(weather_elements[1], labelpad=25, fontsize=15, rotation=270)

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')

            ax2.bar(x_value, y_value[-1], color='skyblue', label='3rd data')
            ax2.set_xlabel(xname, labelpad=5, fontsize=15)
            ax2.set_ylabel(weather_elements[-1], labelpad=15, fontsize=15)
            ax2.tick_params(axis='x', rotation=45, labelsize=10)
            ax2.legend(loc='upper right')

        elif len(weather_elements) == 4:
            weatherPlot, (ax1, ax2) = plt.subplots(2,1, figsize=(10,15), constrained_layout=True)

            line1 = ax1.plot(x_value, y_value[0], color='red', marker = "o", label='1st data')
            ax1.set_xlabel(xname, labelpad=5, fontsize=15)
            ax1.set_ylabel(weather_elements[0], labelpad=15, fontsize=15)
            ax1.set_title("2022 Weather Graph", fontsize=20, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45, labelsize=10)

            ax1_sub = ax1.twinx()
            line2 = ax1_sub.plot(x_value, y_value[1], color='blue', marker = "o", label='2nd data')
            ax1_sub.set_ylabel(weather_elements[1], labelpad=25, fontsize=15, rotation=270)

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')

            ax2.bar(x_value, y_value[-1], color='skyblue', label='3rd data')
            ax2.set_xlabel(xname, labelpad=5, fontsize=15)
            ax2.set_ylabel(weather_elements[-1], labelpad=15, fontsize=15)
            ax2.tick_params(axis='x', rotation=45, labelsize=10)
            ax2.legend(loc='upper left')

            ax2_sub = ax2.twinx()
            ax2_sub.plot(x_value, y_value[2], color='gray', marker = "o", label='4th data')
            ax2_sub.set_ylabel(weather_elements[2], labelpad=25, fontsize=15, rotation=270)
            ax2_sub.legend(loc='upper right')

      
    else:
        if len(weather_elements) == 1:
            weatherPlot = plt.figure(figsize=(10,10))
            plt.title("2022 Weather Graph", fontsize=20, fontweight='bold')
            plt.xlabel(xname,labelpad=5, fontsize=15)
            plt.ylabel(weather_elements[0], labelpad=15, fontsize=15)
            plt.xticks(size=10, rotation=45)
            plt.plot(x_value, y_value[0], color='red', marker='o', label='1st data')
            plt.legend(loc = "upper left")

        elif len(weather_elements) == 2:
            weatherPlot, ax1 = plt.subplots(figsize=(10,10))
            plt.title("2022 Weather Graph", fontsize=20, fontweight='bold')
            plt.xticks(size=10, rotation=45)

            line1 = ax1.plot(x_value, y_value[0], color='red', marker='o', label='1st data')
            ax1.set_xlabel(xname, labelpad=5, fontsize=15)
            ax1.set_ylabel(weather_elements[0], labelpad=15, fontsize=15)

            ax1_sub = ax1.twinx()
            line2 = ax1_sub.plot(x_value, y_value[1], color='skyblue', marker='o', label='2nd data')
            ax1_sub.set_ylabel(weather_elements[1], labelpad=25, fontsize=15, rotation=270)

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
        
        elif len(weather_elements) == 3:
            weatherPlot, (ax1, ax2) = plt.subplots(2,1, figsize=(10,15), constrained_layout=True)

            line1 = ax1.plot(x_value, y_value[0], color='red', marker = "o", label='1st data')
            ax1.set_xlabel(xname, labelpad=5, fontsize=15)
            ax1.set_ylabel(weather_elements[0], labelpad=15, fontsize=15)
            ax1.set_title("2022 Weather Graph", fontsize=20, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45, labelsize=10)

            ax1_sub = ax1.twinx()
            line2 = ax1_sub.plot(x_value, y_value[1], color='skyblue', marker = "o", label='2nd data')
            ax1_sub.set_ylabel(weather_elements[1], labelpad=25, fontsize=15, rotation=270)

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')

            ax2.plot(x_value, y_value[2], color='gray', marker = "o", label='3rd data')
            ax2.set_xlabel(xname, labelpad=5, fontsize=15)
            ax2.set_ylabel(weather_elements[2], labelpad=15, fontsize=15)
            ax2.tick_params(axis='x', rotation=45, labelsize=10)
            ax2.legend(loc='upper left')

    return [weatherTable, weatherPlot]
```
## - Demo Usage

Before you implement demo, you have to install gradio, pandas, and matplotlib library.

```bash
pip install gradio pandas matplotlib
```

Run the code below as a Python script or in a Jupyter Notebook or google colab.

We provided demo code as a .ipynb file.

```python
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

# search weather data that is in csv file
def dataSearch(month, day, weather_elements, location, precipitation):
    if location=='Seoul':
        df = pd.read_csv('Seoul.csv')
    elif location=='Washington':
        df = pd.read_csv('Washington.csv')

    if precipitation:
        weather_elements.append('precipitation')

    if day in ['1','2','3','4','5','6','7','8','9']:
        today = '2022-'+month +'-0'+day
    else:
        today = '2022-'+month+'-'+day

    df1 = df[df.date == today]
    columns = ['location', 'date', 'time'] + weather_elements
    df2 = df1.loc[:, columns]
  
    return df2

# show weather data in plot using matplotlib
def showOutput(month, day, weather_elements, location, precipitation):
    if month=='January':
        month = '01'
    elif month=='February':
        month = '02'
    elif month=='March':
        month = '03'
    elif month=='April':
        month = '04'
    elif month=='May':
        month = '05'
    elif month=='June':
        month = '06'
    elif month=='July':
        month = '07'
    elif month=='August':
        month = '08'
    elif month=='September':
        month = '09'
    elif month=='October':
        month = '10'
    elif month=='November':
        month = '11'
    elif month=='December':
        month = '12'

    weatherTable = dataSearch(month, day, weather_elements, location, precipitation)

    if precipitation:
        weather_elements.remove('precipitation')

    if day in ['1','2','3','4','5','6','7','8','9']:
        xname = '2022-'+month +'-0'+day
    else:
        xname = '2022-'+month+'-'+day

    y_value=[0]*len(weather_elements)

    for i in range(len(weather_elements)):
        y_value[i] = weatherTable[weather_elements[i]]

    x_value = weatherTable['time']

    if 'humidity' in weather_elements:
        humidity_index = weather_elements.index('humidity')
        if weather_elements[humidity_index] != weather_elements[-1]:
            temp = weather_elements[humidity_index]
            weather_elements[humidity_index] = weather_elements[-1]
            weather_elements[-1] = temp


        if len(weather_elements) == 1:
            weatherPlot = plt.figure(figsize=(10,10))
            plt.title("2022 Weather Graph", fontsize=20, fontweight='bold')
            plt.xlabel(xname,labelpad=5, fontsize=15)
            plt.ylabel(weather_elements[0], labelpad=15, fontsize=15)
            plt.xticks(size=10, rotation=45)

            plt.bar(x_value, y_value[-1], color='skyblue', label='1st data')
            plt.legend(loc = "upper left")

        elif len(weather_elements) == 2:
            weatherPlot, ax1 = plt.subplots(figsize=(10,10))
            plt.title("2022 Weather Graph", fontsize=20, fontweight='bold')
            plt.xticks(size=10, rotation=45)

            ax1.bar(x_value, y_value[-1], color='skyblue', label='1st data')
            ax1.set_xlabel(xname, labelpad=5, fontsize=15)
            ax1.set_ylabel(weather_elements[1], labelpad=15, fontsize=15)
            ax1.legend(loc='upper left')

            ax1_sub = ax1.twinx()
            ax1_sub.plot(x_value, y_value[0], color='red', marker = "o", label='2nd data')
            ax1_sub.set_ylabel(weather_elements[0], labelpad=25, fontsize=15, rotation=270)
            ax1_sub.legend(loc='upper right')
            
        elif len(weather_elements) == 3:
            weatherPlot, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,15), constrained_layout=True)

            line1 = ax1.plot(x_value, y_value[0], color='red', marker = "o", label='1st data')
            ax1.set_xlabel(xname, labelpad=5, fontsize=15)
            ax1.set_ylabel(weather_elements[0], labelpad=15, fontsize=15)
            ax1.set_title("2022 Weather Graph", fontsize=20, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45, labelsize=10)

            ax1_sub = ax1.twinx()
            line2 = ax1_sub.plot(x_value, y_value[1], color='blue', marker = "o", label='2nd data')
            ax1_sub.set_ylabel(weather_elements[1], labelpad=25, fontsize=15, rotation=270)

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')

            ax2.bar(x_value, y_value[-1], color='skyblue', label='3rd data')
            ax2.set_xlabel(xname, labelpad=5, fontsize=15)
            ax2.set_ylabel(weather_elements[-1], labelpad=15, fontsize=15)
            ax2.tick_params(axis='x', rotation=45, labelsize=10)
            ax2.legend(loc='upper right')

        elif len(weather_elements) == 4:
            weatherPlot, (ax1, ax2) = plt.subplots(2,1, figsize=(10,15), constrained_layout=True)

            line1 = ax1.plot(x_value, y_value[0], color='red', marker = "o", label='1st data')
            ax1.set_xlabel(xname, labelpad=5, fontsize=15)
            ax1.set_ylabel(weather_elements[0], labelpad=15, fontsize=15)
            ax1.set_title("2022 Weather Graph", fontsize=20, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45, labelsize=10)

            ax1_sub = ax1.twinx()
            line2 = ax1_sub.plot(x_value, y_value[1], color='blue', marker = "o", label='2nd data')
            ax1_sub.set_ylabel(weather_elements[1], labelpad=25, fontsize=15, rotation=270)

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')

            ax2.bar(x_value, y_value[-1], color='skyblue', label='3rd data')
            ax2.set_xlabel(xname, labelpad=5, fontsize=15)
            ax2.set_ylabel(weather_elements[-1], labelpad=15, fontsize=15)
            ax2.tick_params(axis='x', rotation=45, labelsize=10)
            ax2.legend(loc='upper left')

            ax2_sub = ax2.twinx()
            ax2_sub.plot(x_value, y_value[2], color='gray', marker = "o", label='4th data')
            ax2_sub.set_ylabel(weather_elements[2], labelpad=25, fontsize=15, rotation=270)
            ax2_sub.legend(loc='upper right')

      
    else:
        if len(weather_elements) == 1:
            weatherPlot = plt.figure(figsize=(10,10))
            plt.title("2022 Weather Graph", fontsize=20, fontweight='bold')
            plt.xlabel(xname,labelpad=5, fontsize=15)
            plt.ylabel(weather_elements[0], labelpad=15, fontsize=15)
            plt.xticks(size=10, rotation=45)
            plt.plot(x_value, y_value[0], color='red', marker='o', label='1st data')
            plt.legend(loc = "upper left")

        elif len(weather_elements) == 2:
            weatherPlot, ax1 = plt.subplots(figsize=(10,10))
            plt.title("2022 Weather Graph", fontsize=20, fontweight='bold')
            plt.xticks(size=10, rotation=45)

            line1 = ax1.plot(x_value, y_value[0], color='red', marker='o', label='1st data')
            ax1.set_xlabel(xname, labelpad=5, fontsize=15)
            ax1.set_ylabel(weather_elements[0], labelpad=15, fontsize=15)

            ax1_sub = ax1.twinx()
            line2 = ax1_sub.plot(x_value, y_value[1], color='skyblue', marker='o', label='2nd data')
            ax1_sub.set_ylabel(weather_elements[1], labelpad=25, fontsize=15, rotation=270)

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
        
        elif len(weather_elements) == 3:
            weatherPlot, (ax1, ax2) = plt.subplots(2,1, figsize=(10,15), constrained_layout=True)

            line1 = ax1.plot(x_value, y_value[0], color='red', marker = "o", label='1st data')
            ax1.set_xlabel(xname, labelpad=5, fontsize=15)
            ax1.set_ylabel(weather_elements[0], labelpad=15, fontsize=15)
            ax1.set_title("2022 Weather Graph", fontsize=20, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45, labelsize=10)

            ax1_sub = ax1.twinx()
            line2 = ax1_sub.plot(x_value, y_value[1], color='skyblue', marker = "o", label='2nd data')
            ax1_sub.set_ylabel(weather_elements[1], labelpad=25, fontsize=15, rotation=270)

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')

            ax2.plot(x_value, y_value[2], color='gray', marker = "o", label='3rd data')
            ax2.set_xlabel(xname, labelpad=5, fontsize=15)
            ax2.set_ylabel(weather_elements[2], labelpad=15, fontsize=15)
            ax2.tick_params(axis='x', rotation=45, labelsize=10)
            ax2.legend(loc='upper left')

    return [weatherTable, weatherPlot]

output1 = gr.Dataframe()
output2 = gr.Plot()

# make gradio interface
demo = gr.Interface(
    fn=showOutput,
    inputs=[
        gr.Dropdown(["January", "February", "March", "April", "May","June",
                     "July", "August", "September", "October", "November", "December"],label="Month", info="Select Months"),
        gr.Dropdown(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                     "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                     "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"], label="Day", info="Select Day"),
        gr.CheckboxGroup(["temperature(˚C)", "wind(m/s)", "humidity(%)", "air_pressure(hPa)"],
                        label="Weather element", info="Choose weather element"),
        gr.Radio(["Washington", "Seoul"], label="Location", info="Choose location"),
        gr.Checkbox(label="precipation?")],
        outputs=[output1,output2]
)

if __name__=="__main__":
    demo.launch()
```

The demo below will appear automatically within the Jupyter Notebook, or pop in a browser on http://localhost:7860 if running from a script:

![weather data demo](readme_files/screenshot.gif =1200x840)
