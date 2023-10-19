import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, NelsonAalenFitter, WeibullFitter

def import_data():
    path = "/Users/patricksweeney/Desktop/wtp.xlsx"
    df = pd.read_excel(path)
    data = df.iloc[:, 1].values
    return data

def kaplan_survival_function(data):
    kmf = KaplanMeierFitter()
    kmf.fit(data)
    return kmf

def nelson_cum_hazard_rate(data):
    naf = NelsonAalenFitter()
    naf.fit(data)
    return naf

def weibull_survival_function(data):
    wf = WeibullFitter()
    wf.fit(data)
    return wf

def weibull_cum_hazard_rate(data):
    wf = WeibullFitter()
    wf.fit(data)
    return wf.cumulative_hazard_

def plot_all(kmf, naf, wf):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    kmf.plot()
    plt.title('Kaplan-Meier Survival Function')
    
    plt.subplot(2, 2, 2)
    naf.plot()
    plt.title('Nelson-Aalen Cumulative Hazard Rate')
    
    plt.subplot(2, 2, 3)
    wf.plot()
    plt.title('Weibull Survival Function')
    
    plt.subplot(2, 2, 4)
    plt.plot(wf.cumulative_hazard_)
    plt.title('Weibull Cumulative Hazard Rate')
    
    plt.show()

# Example usage
data = import_data()
kmf = kaplan_survival_function(data)
naf = nelson_cum_hazard_rate(data)
wf = weibull_survival_function(data)

plot_all(kmf, naf, wf)
