import numpy as np
import math


def GMM(alpha, x, u, conv,dim):
    covdet = np.linalg.det(conv + np.eye(dim) * 0.001)
    covinv = np.linalg.inv(conv + np.eye(dim) * 0.001)
    T1 = 1 / ( (2 * math.pi)**(dim/2) * np.sqrt(covdet))
    T2 = np.exp((-0.5) * ((np.transpose(x - u)).dot(covinv).dot(x - u)))
    prob = T1 * T2
    return alpha * prob[0]

def EM_GMM(weights,mean,cov,data,M,dim):
    initial_value = 0
    for i in data:
        i = np.expand_dims(i, 1)
        all_value = 0
        for k in range(M):
            value = GMM(weights[k], i, mean[k], cov[k],dim)
            all_value = all_value + value

        intial_value_temp = math.log(all_value + 0.00001)
        initial_value = initial_value + intial_value_temp

    flag = 10000
    num = 0
    while (flag > 0.00001):
        print("flag",flag)
        num = num + 1
        P = []
        for m in range(M):
            l = [] * (m + 1)
            P.append(l)

        # E step
        for i in data:
            i = np.reshape(i, (dim, 1))

            value = [GMM(weights[k], i, mean[k], cov[k],dim) for k in range(M)]
            value = np.array(value)
            sum_value = np.sum(value)

            for m in range(M):
                p = GMM(weights[m], i, mean[m], cov[m],dim) / sum_value
                P[m].append(p)

        for m in range(M):
            P[m] = np.array(P[m])  # 1000*1

        # M step
        # update alpha
        for m in range(M):
            weights[m] = np.sum(P[m]) / len(data)

        # update u
        for m in range(M):
            result_list = []
            for i in range(len(data)):
                W = np.expand_dims(data[i], 1)
                result = P[m][i] * W
                result_list.append(result)
            result_list = np.array(result_list)
            mean_sum = np.sum(result_list, 0)
            mean[m] = mean_sum / np.sum(P[m])

        # update cov
        for m in range(M):
            result_list = []
            for i in range(len(data)):
                W = np.expand_dims(data[i], 1)  # 2 * 1
                T = W - mean[m]
                Q = np.transpose(T)
                temp = (T.dot(Q)) * P[m][i]
                result_list.append(temp)
            result_list = np.array(result_list)
            cov_sum = np.sum(result_list, 0)
            cov[m] = cov_sum / np.sum(P[m])

        update_value = 0
        for i in data:
            i = np.expand_dims(i, 1)
            all_value = 0
            for k in range(M):
                value = GMM(weights[k], i, mean[k], cov[k],dim)
                all_value = all_value + value

            update_value_temp = math.log(all_value)
            update_value = update_value + update_value_temp

        flag = abs(update_value - initial_value)
        initial_value = update_value

    return weights,mean,cov,num