import leather

def dotGraph(fileName, data):
	chart = leather.Chart('Dots')
	chart.add_dots(data)
	chart.to_svg(fileName)