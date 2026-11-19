import matplotlib.pyplot as plt

cities       = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"]
population   = [32, 21, 14, 11, 15]
rainfall     = [800, 2400, 970, 1400, 1600]
temperatures = [32, 33, 28, 31, 30]

plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.bar(cities, population, color="purple")
plt.title("Population (millions)")
plt.xticks(rotation=30)

plt.subplot(1, 3, 2)
plt.plot(cities, rainfall, marker="o", color="blue")
plt.title("Rainfall (mm)")
plt.xticks(rotation=30)
plt.grid(True)

plt.subplot(1, 3, 3)
plt.hist(temperatures, bins=4, color="orange", edgecolor="black")
plt.title("Temperature Histogram")
plt.xlabel("Temperature (\u00b0C)")

plt.tight_layout()
plt.show()

print("Highest population :", cities[population.index(max(population))])
print("Highest rainfall   :", cities[rainfall.index(max(rainfall))])
print("Lowest rainfall    :", cities[rainfall.index(min(rainfall))])
print("Hottest city       :", cities[temperatures.index(max(temperatures))])
print("Coolest city       :", cities[temperatures.index(min(temperatures))])
print("Average temperature:", sum(temperatures) / len(temperatures))

print("""
REPORT
Delhi has the largest population (32 million) but the lowest rainfall (800 mm).
Mumbai has the highest rainfall (2400 mm) and the highest temperature (33 C).
Bangalore is the coolest city (28 C) with moderate rainfall (970 mm).
Coastal cities (Mumbai, Chennai, Kolkata) get much more rain than inland Delhi.
Temperatures are close across cities (28-33 C, average 30.8 C), so rainfall
is the main climatic difference between them.
""")