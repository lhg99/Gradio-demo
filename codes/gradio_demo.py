# This file is app.py hosted on huggingface.

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
    weather_elements.append('precipitation(mm)')
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
    weather_elements.remove('precipitation(mm)')

  if day in ['1','2','3','4','5','6','7','8','9']:
        xname = '2022-'+month +'-0'+day
  else:
        xname = '2022-'+month+'-'+day

  y_value=[0]*len(weather_elements)

  x_value = weatherTable['time']

  if 'humidity(%)' in weather_elements:
    humidity_index = weather_elements.index('humidity(%)')
    if weather_elements[humidity_index] != weather_elements[-1]:
      temp = weather_elements[humidity_index]
      weather_elements[humidity_index] = weather_elements[-1]
      weather_elements[-1] = temp
    for i in range(len(weather_elements)):
      y_value[i] = weatherTable[weather_elements[i]]


    if len(weather_elements) == 1:
      weatherPlot = plt.figure(figsize=(10,10))
      plt.title("2022 Weather Graph", fontsize=20, fontweight='bold')
      plt.xlabel(xname,labelpad=5, fontsize=15)
      plt.ylabel(weather_elements[0], labelpad=15, fontsize=15)
      plt.xticks(size=10, rotation=45)

      plt.bar(x_value, y_value[-1], color='skyblue', label=weather_elements[0])
      plt.legend(loc = "upper left")

    elif len(weather_elements) == 2:
      weatherPlot, ax1 = plt.subplots(figsize=(10,10))
      plt.title("2022 Weather Graph", fontsize=20, fontweight='bold')
      plt.xticks(size=10, rotation=45)

      ax1.bar(x_value, y_value[-1], color='skyblue', label=weather_elements[1])
      ax1.set_xlabel(xname, labelpad=5, fontsize=15)
      ax1.set_ylabel(weather_elements[1], labelpad=15, fontsize=15)
      ax1.legend(loc='upper left')

      ax1_sub = ax1.twinx()
      ax1_sub.plot(x_value, y_value[0], color='red', marker = "o", label=weather_elements[0])
      ax1_sub.set_ylabel(weather_elements[0], labelpad=25, fontsize=15, rotation=270)
      ax1_sub.legend(loc='upper right')
      
    elif len(weather_elements) == 3:
      weatherPlot, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,15), constrained_layout=True)

      line1 = ax1.plot(x_value, y_value[0], color='red', marker = "o", label=weather_elements[0])
      ax1.set_xlabel(xname, labelpad=5, fontsize=15)
      ax1.set_ylabel(weather_elements[0], labelpad=15, fontsize=15)
      ax1.set_title("2022 Weather Graph", fontsize=20, fontweight='bold')
      ax1.tick_params(axis='x', rotation=45, labelsize=10)

      ax1_sub = ax1.twinx()
      line2 = ax1_sub.plot(x_value, y_value[1], color='blue', marker = "o", label=weather_elements[1])
      ax1_sub.set_ylabel(weather_elements[1], labelpad=25, fontsize=15, rotation=270)

      lines = line1 + line2
      labels = [l.get_label() for l in lines]
      ax1.legend(lines, labels, loc='upper left')

      ax2.bar(x_value, y_value[-1], color='skyblue', label=weather_elements[-1])
      ax2.set_xlabel(xname, labelpad=5, fontsize=15)
      ax2.set_ylabel(weather_elements[-1], labelpad=15, fontsize=15)
      ax2.tick_params(axis='x', rotation=45, labelsize=10)
      ax2.legend(loc='upper right')

    elif len(weather_elements) == 4:
      weatherPlot, (ax1, ax2) = plt.subplots(2,1, figsize=(10,15), constrained_layout=True)

      line1 = ax1.plot(x_value, y_value[0], color='red', marker = "o", label=weather_elements[0])
      ax1.set_xlabel(xname, labelpad=5, fontsize=15)
      ax1.set_ylabel(weather_elements[0], labelpad=15, fontsize=15)
      ax1.set_title("2022 Weather Graph", fontsize=20, fontweight='bold')
      ax1.tick_params(axis='x', rotation=45, labelsize=10)

      ax1_sub = ax1.twinx()
      line2 = ax1_sub.plot(x_value, y_value[1], color='blue', marker = "o", label=weather_elements[1])
      ax1_sub.set_ylabel(weather_elements[1], labelpad=25, fontsize=15, rotation=270)

      lines = line1 + line2
      labels = [l.get_label() for l in lines]
      ax1.legend(lines, labels, loc='upper left')

      ax2.bar(x_value, y_value[-1], color='skyblue', label=weather_elements[-1])
      ax2.set_xlabel(xname, labelpad=5, fontsize=15)
      ax2.set_ylabel(weather_elements[-1], labelpad=15, fontsize=15)
      ax2.tick_params(axis='x', rotation=45, labelsize=10)
      ax2.legend(loc='upper left')

      ax2_sub = ax2.twinx()
      ax2_sub.plot(x_value, y_value[2], color='gray', marker = "o", label=weather_elements[2])
      ax2_sub.set_ylabel(weather_elements[2], labelpad=25, fontsize=15, rotation=270)
      ax2_sub.legend(loc='upper right')

      
  else:
    for i in range(len(weather_elements)):
      y_value[i] = weatherTable[weather_elements[i]]

    if len(weather_elements) == 1:
      weatherPlot = plt.figure(figsize=(10,10))
      plt.title("2022 Weather Graph", fontsize=20, fontweight='bold')
      plt.xlabel(xname,labelpad=5, fontsize=15)
      plt.ylabel(weather_elements[0], labelpad=15, fontsize=15)
      plt.xticks(size=10, rotation=45)
      plt.plot(x_value, y_value[0], color='red', marker='o', label=weather_elements[0])
      plt.legend(loc = "upper left")

    elif len(weather_elements) == 2:
      weatherPlot, ax1 = plt.subplots(figsize=(10,10))
      plt.title("2022 Weather Graph", fontsize=20, fontweight='bold')
      plt.xticks(size=10, rotation=45)

      line1 = ax1.plot(x_value, y_value[0], color='red', marker='o', label=weather_elements[0])
      ax1.set_xlabel(xname, labelpad=5, fontsize=15)
      ax1.set_ylabel(weather_elements[0], labelpad=15, fontsize=15)

      ax1_sub = ax1.twinx()
      line2 = ax1_sub.plot(x_value, y_value[1], color='skyblue', marker='o', label=weather_elements[1])
      ax1_sub.set_ylabel(weather_elements[1], labelpad=25, fontsize=15, rotation=270)

      lines = line1 + line2
      labels = [l.get_label() for l in lines]
      ax1.legend(lines, labels, loc='upper left')
      
    elif len(weather_elements) == 3:
      weatherPlot, (ax1, ax2) = plt.subplots(2,1, figsize=(10,15), constrained_layout=True)

      line1 = ax1.plot(x_value, y_value[0], color='red', marker = "o", label=weather_elements[0])
      ax1.set_xlabel(xname, labelpad=5, fontsize=15)
      ax1.set_ylabel(weather_elements[0], labelpad=15, fontsize=15)
      ax1.set_title("2022 Weather Graph", fontsize=20, fontweight='bold')
      ax1.tick_params(axis='x', rotation=45, labelsize=10)

      ax1_sub = ax1.twinx()
      line2 = ax1_sub.plot(x_value, y_value[1], color='skyblue', marker = "o", label=weather_elements[1])
      ax1_sub.set_ylabel(weather_elements[1], labelpad=25, fontsize=15, rotation=270)

      lines = line1 + line2
      labels = [l.get_label() for l in lines]
      ax1.legend(lines, labels, loc='upper left')

      ax2.plot(x_value, y_value[2], color='gray', marker = "o", label=weather_elements[2])
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
        gr.Dropdown(["January", "February", "March", "April", "May", "June",
                     "July", "August", "September", "October", "November", "December"],
                     label="Month", info="Select Months"),
        gr.Dropdown(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                     "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                     "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"], label="Day", info="Select Day"),
        gr.CheckboxGroup(["temperature(˚C)", "wind(m/s)", "humidity(%)", "air_pressure(hPa)"],
                         label="Weather element", info="Choose weather element"),
        gr.Radio(["Washington", "Seoul"],
                 label="Location", info="Choose location"),
        gr.Checkbox(label="precipation?")],
        outputs=[output1,output2]
)

if __name__=="__main__":
  demo.launch()