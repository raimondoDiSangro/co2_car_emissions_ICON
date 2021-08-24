import pandas as pd
import tkinter as tk
from tkinter import messagebox
from tkinter.scrolledtext import *
from source.models import svr_model, decision_tree_model, rfr_model
from source.clustering import clusterkMeans

cars_cat_kmeans = ['make', 'model', 'engine_size', 'cylinders',
                   'fuel_consumption_city', 'fuel_consumption_hwy', 'fuel_consumption_comb', 'co2_emissions']


class Dialogue(tk.Frame):
    def __init__(self):
        self.engine_size = 0
        self.cylinders = 0
        self.fuel_consumption_city = 0
        self.fuel_consumption_hwy = 0
        self.fuel_consumption_comb = 0
        self.cars_df = pd.read_csv('../data/co2_emissions.csv')

        tk.Frame.__init__(self)
        self.master.title("Car co2 emissions machine learning")
        self.master.minsize(550, 450)
        self.grid(sticky=tk.E + tk.W + tk.N + tk.S)

        # resizable window
        # top = self.winfo_toplevel()
        # top.rowconfigure(0, weight=1)
        # top.columnconfigure(0, weight=1)

        for i in range(12):
            self.rowconfigure(i, weight=1)
            self.columnconfigure(1, weight=1)

        self.label1 = tk.Label(self, text="Engine size in liters (es 1.4):")  # engine size label
        self.label1.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)

        self.input1 = tk.Entry(self)  # input box for the engine size
        self.input1.grid(column=1, row=0, sticky=tk.E, padx=5, pady=5)

        self.label2 = tk.Label(self, text="Number of cylinders (es. 4):")  # cylinders label
        self.label2.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)

        self.input2 = tk.Entry(self)  # input cylinders
        self.input2.grid(column=1, row=1, sticky=tk.E, padx=5, pady=5)

        self.label3 = tk.Label(self, text="Fuel Consumption City: (in l/100km)")  # fuel_consumption_city label
        self.label3.grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)

        self.input3 = tk.Entry(self)  # fuel_consumption_city
        self.input3.grid(column=1, row=2, sticky=tk.E, padx=5, pady=5)

        self.label4 = tk.Label(self, text="Fuel Consumption highway: (in l/100km)")  # fuel_consumption_hwy label
        self.label4.grid(column=0, row=3, sticky=tk.W, padx=5, pady=5)

        self.input4 = tk.Entry(self)  # fuel_consumption_hwy input
        self.input4.grid(column=1, row=3, sticky=tk.E, padx=5, pady=5)

        self.label5 = tk.Label(self, text="Fuel Consumption Comb: (in l/100km)")  # fuel_consumption_comb label
        self.label5.grid(column=0, row=4, sticky=tk.W, padx=5, pady=5)

        self.input5 = tk.Entry(self)  # fuel_consumption_comb input
        self.input5.grid(column=1, row=4, sticky=tk.E, padx=5, pady=5)

        self.label6 = tk.Label(self, text="Predicted Emissions ->")  # label of emissions
        self.label6.grid(column=0, row=5, sticky=tk.W, padx=5, pady=5)

        self.label6 = tk.Label(self, text="Similar cars ->")  # label similar cars
        self.label6.grid(column=0, row=7, sticky=tk.W, padx=5, pady=5)

        self.button1 = tk.Button(self, text="Prediction", command=self.input_compute)
        self.button1.grid(column=2, row=3, sticky=tk.E, padx=5, pady=5)
        self.button2 = tk.Button(self, text="Reset", command=self.reset)
        self.button2.grid(column=2, row=1, sticky=tk.E, padx=5, pady=5)

        self.regression_result = tk.Text(self, height=10, width=50)
        self.regression_result.grid(column=1, row=5, sticky=tk.E, padx=5, pady=5)

        # self.kmeans_result = tk.Text(self, height=20, width=50)
        self.kmeans_result = ScrolledText(self, height=20, width=48)

        self.kmeans_result.grid(column=1, row=7, sticky=tk.E, padx=5, pady=5)

    def input_compute(self):
        message = 0
        input_check = 1
        input_data = []  #

        self.regression_result.delete("1.0", "end")
        self.kmeans_result.delete("1.0", "end")
        # self.kmeans_result.delete("1.0", "end")
        # engine_size,cylinders,fuel_consumption_city, fuel_consumption_hwy,fuel_consumption_comb

        if len(self.input1.get()) != 0:
            input_check = self.checkInput("engine_size", self.input1.get())
        else:
            message = 1
        if (input_check == 0) and (message == 0):
            self.engine_size = float(self.input1.get())
            input_data.append(self.engine_size)

        if len(self.input2.get()) != 0:
            input_check = self.checkInput("cylinders", self.input2.get())
        else:
            message = 1
        if (input_check == 0) and (message == 0):
            self.cylinders = float(self.input2.get())
            input_data.append(self.cylinders)

        if len(self.input3.get()) != 0:
            input_check = self.checkInput("fuel_consumption_city", self.input3.get())
        else:
            message = 1
        if (input_check == 0) and (message == 0):
            self.fuel_consumption_city = float(self.input3.get())
            input_data.append(self.fuel_consumption_city)

        if len(self.input4.get()) != 0:
            input_check = self.checkInput("fuel_consumption_hwy", self.input4.get())
        else:
            message = 1
        if (input_check == 0) and (message == 0):
            self.fuel_consumption_hwy = float(self.input4.get())
            input_data.append(self.fuel_consumption_hwy)

        if len(self.input5.get()) != 0:
            input_check = self.checkInput("fuel_consumption_comb", self.input5.get())
        else:
            message = 1
        if (input_check == 0) and (message == 0):
            self.fuel_consumption_comb = float(self.input5.get())
            input_data.append(self.fuel_consumption_comb)
        else:
            message = 1

        if (message == 1) and (input_check == 0):
            messagebox.showerror("Error", "Please insert all the values required")

        # cars_cat_kmeans = ['make', 'model', 'engine_size', 'cylinders', 'fuel_consumption_city',
        # 'fuel_consumption_hwy', 'fuel_consumption_comb']

        # values = {'engine_size': self.engine_size, 'cylinders': self.cylinders, 'fuel_consumption_city':
        # self.fuel_consumption_city, 'fuel_consumption_hwy': self.fuel_consumption_hwy, 'fuel_consumption_comb':
        # self.fuel_consumption_comb}

        if (input_check == 0) and (message == 0):
            # svr_result = str(svr_model(self.cars_df, input_data))
            # self.cars_df = pd.read_csv('../data/co2_emissions.csv')

            # float_svr_result = (svr_model(self.cars_df, input_data))
            # self.cars_df = pd.read_csv('../data/co2_emissions.csv')

            # decision_tree_result = str(decision_tree_model(self.cars_df, input_data))
            # self.cars_df = pd.read_csv('../data/co2_emissions.csv')

            random_forest_result = str(rfr_model(self.cars_df, input_data))
            self.cars_df = pd.read_csv('../data/co2_emissions.csv')

            #float_forest_result = (rfr_model(self.cars_df, input_data))
            float_forest_result = int(random_forest_result)
            self.cars_df = pd.read_csv('../data/co2_emissions.csv')

            # self.regression_result.insert(tk.END, "Svr result: " + svr_result + "g/km\n")
            # self.regression_result.insert(tk.END, "Decision Tree result:" + decision_tree_result + "g/km\n")
            self.regression_result.insert(tk.END, "Forest Tree result: " + random_forest_result + "g/km\n")

            values = {'engine_size': self.engine_size, 'cylinders': self.cylinders,
                      'fuel_consumption_city': self.fuel_consumption_city,
                      'fuel_consumption_hwy': self.fuel_consumption_hwy,
                      'fuel_consumption_comb': self.fuel_consumption_comb,
                      # 'co2_emissions': float_svr_result
                      'co2_emissions': float_forest_result}

            # print(result)
            # self.cars_df = pd.read_csv('../data/co2_emissions.csv')
            result_kmeans = clusterkMeans(self.cars_df, cars_cat_kmeans, values)

            if len(result_kmeans) == 0:  # prints an error message if there are no similar cars in the dataset
                self.kmeans_result.insert(tk.END, "No Similar cars found in the dataset,\nplease insert new data")
            else:  # stampo i risultati
                self.kmeans_result.insert(tk.END, result_kmeans)
                self.cars_df = pd.read_csv('../data/co2_emissions.csv')

            self.cars_df = pd.read_csv('../data/co2_emissions.csv')

    def checkInput(self, feature, number):
        check = 0
        message = 0

        if feature == "engine_size":
            try:
                float(number)
                if float(number) > 10 or float(number) < 0.5:
                    self.reset()
                    message = 1
                    check = 1
            except ValueError:
                self.reset()
                message = 2
                check = 1

        if feature == "cylinders":
            try:
                float(number)
                if float(number) > 16 or float(number) < 1:
                    self.reset()
                    message = 1
                    check = 1
            except ValueError:
                self.reset()
                message = 2
                check = 1

        if (feature == "fuel city" or feature == "fuel hwy"
                or feature == "fuel comb"):
            try:
                float(number)
                if float(number) > 99 or float(number) < 0:
                    self.reset()
                    message = 1
                    check = 1
            except ValueError:
                self.reset()
                message = 2
                check = 1

        if message == 1:
            messagebox.showerror("INVALID INPUT",
                                 "Please insert suitable values")
        elif message == 2:
            messagebox.showerror("INVALID INPUT", "Please insert only numerical values")
        return check

    def reset(self):
        self.regression_result.delete("1.0", "end")
        self.kmeans_result.delete("1.0", "end")
        self.input1.delete(0, "end")
        self.input2.delete(0, "end")
        self.input3.delete(0, "end")
        self.input4.delete(0, "end")
        self.input5.delete(0, "end")
        self.cars_df = pd.read_csv('../data/co2_emissions.csv')


if __name__ == "__main__":
    d = Dialogue()
    d.mainloop()
