import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors as ns


def distance(xt1, xt2, x11, x12):
    return (abs(xt1 - x11) ** 2 + abs(xt2 - x12) ** 2) ** 0.5

# dataset = [['Продукт', 'Сладость', 'Хруст', 'Класс'],
#             ['яблоко', '7', '7', '0'],
#             ['салат', '2', '5', '1'],
#             ['бекон', '1', '2', '2'],
#             ['круассан', '5', '1', '3'],
#             ['банан', '9', '1', '0'],
#             ['орехи', '1', '5', '2'],
#             ['рыба',	'1', '1', '2'],
#             ['сухари', '6', '5', '3'],
#             ['пряники',	'4', '2', '3'],
#             ['сыр',	'1', '1', '2'],
#             ['виноград',	'8', '1', '0'],
#             ['апельсин',	'6', '1', '0'],                    
#             ['печенье', '5', '4', '3'],
#             ['морковь',	'2', '8', '1'],
#             ['апельсин',	'6', '1', '0'],
#             ['гранат', '7', '3', '0'],
#             ['кекс', '5', '1', '3'],
#             ['помидор', '2', '1', '1'],
#             ['грудка', '1', '2', '2'],
#             ['арбуз', '8', '5', '0'],
#             ['икра',	'1', '1', '2'],
#             ['мандарин',	'9', '1', '0']]

# with open('lab4_data.csv', 'w', newline='') as f:
#     writer = csv.writer(f);
#     writer.writerows(dataset);

data = pd.read_csv('lab4_data.csv', encoding='cp1251');

new_dist = np.zeros((11, 11))

for i in range(11):
    for j in range(11):
        new_dist[i][j] = distance(int(data.iloc[11+i][1]), int(data.iloc[11+i][1]), int(data.iloc[j+1][2]), int(data.iloc[j+1][2]))

print('new_dist:')
print(new_dist)

er_k = [0] * 10

for k in range(7):
    print('____________________________________')    
    print('Классификация для k =', k + 1)

    er = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(11):
        qwant_dist = [0, 0, 0, 0]
        print('\n')
        print('  Ближайшие соседи продукта '+str(data.iloc[11 + i][0])+':')     
        tmp = np.array(new_dist[i, :])      

        for j in range(k + 1):        
            ind_min = list(tmp).index(min(tmp))  
                
            print('    ',data.iloc[ind_min][0],'(класс '+str(data.iloc[ind_min + i + 1][3])+')')
        
            qwant_dist[int(data.iloc[ind_min + i + 1][3])] += 1                  
            tmp[ind_min] = 1000

            max1 = max(qwant_dist)

            if qwant_dist.count(max1) > 1:         
                er[i] = 0.5

            elif int(data.iloc[ind_min + i + 1][3]) != int(data.iloc[11 + i][3]):
                er[1] = 1

        er_k[k] = np.mean(er)

print('\nОшибки:', er_k)

plt.plot(range(1, 11), er_k)
plt.title('График ошибки в зависимости от k')
plt.xlabel('k')
plt.ylabel('Ошибка')
plt.show() 

xTrain = data.iloc[:11][['Сладость', 'Хруст']]
xTest = data.iloc[11:][['Сладость', 'Хруст']]
yTrain = data.iloc[:11]['Класс']
yTest = data.iloc[11:]['Класс']

error = []

for i in range(1, 7): 
    knn = ns.KNeighborsClassifier(n_neighbors=i) 
    knn.fit(xTrain, yTrain) 
    pred_i = knn.predict(xTest) 
    error.append(np.mean(pred_i != yTest))

plt.plot(range(1, 7), error) 
plt.title('График ошибки в зависимости от k(sklearn)') 
plt.xlabel('k') 
plt.ylabel('Ошибка')
plt.show()

plt.scatter(data['Сладость'], data['Хруст'])
plt.title('Диаграмма продуктов') 
plt.xlabel('Сладость') 
plt.ylabel('Хруст')
plt.show()

