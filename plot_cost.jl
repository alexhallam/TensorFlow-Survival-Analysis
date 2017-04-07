#libraries
using DataFrames
using Gadfly

#read data
df = readtable("cost_values.csv")

#make a column of index
df = DataFrame(idx = 1:length(df[1]), cost = df[1])

#Gadfly plot
p = plot(df, x=:idx, y=:cost, Geom.point);

#save image
img = SVG("cost_plot.svg", 12inch, 8inch)

#draw image
draw(img, p)
