import numpy
import sys


def run_demo(args):
    print (args)
    data = args[1]
    model = ProductRecommender()
    model.fit(data)


class ProductRecommender(object):
    """
    Example:
    from matrix_factor_model import ProductRecommender
    modelA = ProductRecommender()
    data = [[1,2,3], [0,2,3]]
    modelA.fit(data)
    model.predict_instance(1)
    # prints array([ 0.9053102 ,  2.02257811,  2.97001565])
    """

    def __init__(self):
        """
        定义模型参数：
        在train中定义
        P = 用户 x 隐向量
        Q = 产品 x 隐向量
        
        """
        self.Q = None
        self.P = None
        

    def fit(self, user_x_product, latent_features_guess=2, learning_rate=0.0002, steps=5000, regularization_penalty=0.02, convergeance_threshold=0.001):
        """
        通过下列参数训练模型
        :param: 用户_x_产品:
        :param: 隐向量维度:
        :param: 学习率:
        :param: 迭代数:
        :param: 正则项惩罚系数:
        :param: 收敛阈值
        :return: 调用函数__factor_matrix，
        """
        print ('training model...')
        return self.__factor_matrix(user_x_product, latent_features_guess, learning_rate, steps, regularization_penalty, convergeance_threshold)

    def predict_instance(self, row_index):
        """
        预测给定行的用户物品评分
        :param 
        row_index: 给定的用户行数（用户id）
        :return:
        用户对所有物品的评分预测
        """
        return numpy.dot(self.P[row_index, :], self.Q.T)

    def predict_all(self):
        """
        填充所有的 用户x物品矩阵
        :return:
        预测的 用户x物品矩阵
        """
        return numpy.dot(self.P, self.Q.T)

    def get_models(self):
        """
        返回P，Q矩阵
        :return:
        P： 用户 x 隐向量
        Q： 产品 x 隐向量
        """
        return self.P, self.Q

    def __factor_matrix(self, R, K, alpha, steps, beta, error_limit):
        """
        R = 用户 x 产品矩阵
        K = 隐向量维度 
        alpha = 学习率
        step = 回归迭代数
        beta = 正则项惩罚系数
        error_limit = 收敛阈值

        Returns:
        P = 用户 x 隐向量. (矩阵的每一column都是一个隐含特征)
        Q = 产品 x 隐向量. (矩阵的每一column都是一个隐含特征)
        通过 dot P 和 Q.T 填充对应用户的R矩阵(通过填充缺失值)
        """
        # 将用户 x 产品矩阵转化成np.array
        R = numpy.array(R)

        # N:  用户数量
        # M： 产品数量
        # 随机初始化 用户 x 隐向量矩阵
        N = len(R)
        M = len(R[0])
        P = numpy.random.rand(N, K)

        # 随机初始化 产品 x 隐向量矩阵
        # 转置得到 隐向量 x 产品矩阵（方便dot操作）
        Q = numpy.random.rand(M, K)
        Q = Q.T
        
        # 初始化误差
        error = 0

        # 迭代的最大代数
        for step in range(steps):

            # 迭代矩阵中的每一个元素
            for i in range(len(R)):
                for j in range(len(R[i])):
                    # 如果用户 x 物品矩阵中有记录，计算误差值
                    if R[i][j] > 0:

                        # P矩阵中的i row 和转置后的Q矩阵中的j column内积，同时和原有值之间的difference得到error
                        eij = R[i][j] - numpy.dot(P[i, :], Q[:, j])

                        for k in range(K):
                            # 根据误差更新 用户x隐含向量矩阵 中的用户向量p
                            P[i][k] = P[i][k] + alpha * (eij * Q[k][j] - beta * P[i][k])
                            # 根据误差更新 隐含向量x物品矩阵 中的物品向量向量q
                            Q[k][j] = Q[k][j] + alpha * (eij * P[i][k] - beta * Q[k][j] )

            # 每次迭代计算误差
            error = self.__error(R, P, Q, K, beta)

            # 当误差小于收敛阈值时跳出迭代循环
            if error < error_limit:
                break

      
        # 更新模型对象参数
        # P： 用户 x 隐向量
        # Q： 物品 x 隐向量   
        self.P = P
        self.Q = Q.T
       

        self.__print_fit_stats(error, N, M, step)

    def __error(self, R, P, Q, K, beta):
        """
        计算误差
        :param R: 用户 x 产品矩阵
        :param P: 用户 x 隐向量
        :param Q: 隐向量 x 产品矩阵
        :param K: 隐向量维度
        :param beta: 正则项惩罚系数
        :return: 
        e: 总体误差
        """
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:

                    # error实际样本和预测值差值的平方和
                    e = e + pow(R[i][j]-numpy.dot(P[i,:],Q[:,j]), 2)

                    # 添加正则项
                    for k in range(K):

                        # error + ||P||^2 + ||Q||^2
                        e = e + (beta) * ( pow(P[i][k], 2) + pow(Q[k][j], 2) )
        return e

    def __print_fit_stats(self, error, samples_count, products_count,step):
        print ('training complete...')
        print ('------------------------------')
        print ('Status:')
        print ('Error: %0.2f' % error)
        print ('Iteration: %d' %step)
        print ('Samples: ' + str(samples_count))
        print ('Products: ' + str(products_count))
        print ('------------------------------')

if __name__ == '__main__':
    run_demo(sys.argv)
