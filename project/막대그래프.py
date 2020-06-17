from pandas import DataFrame
from pandas import Series
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
from matplotlib import style
from matplotlib import font_manager, rc

style.use('ggplot')

figure = ['1','2','3', '4', '5',
            '5', '6', '7', '8', '9',
            '11', '12', '13', '14', '15',
            '16', '17', '18', '19', '20',
            '21', '22', '23', '24', '25',
            '26', '27', '28', '29', '30',
            '31', '32', '33', '34', '35',
            '36', '37', '38', '39', '40',
            '41', '42', '43', '44', '45']


number = [128, 120, 119, 124, 127, 115, 121, 124, 93, 125, 
                124, 130, 128, 130, 122, 120, 132, 132, 128, 128,
                123, 102, 108, 121, 118, 124, 135, 114, 111, 111,
                123, 106, 127, 143, 112, 119, 127, 122, 131, 132,
                110, 118, 133, 120, 130]

fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111)

ypos = np.arange(45)
rects = plt.barh(ypos, number, align='center', height=0.4)
plt.yticks(ypos, figure)

plt.title('frequency')

plt.tight_layout()
plt.show()