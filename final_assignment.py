import numpy as np
class matrix:
    '''Implementing the matrix class'''
    def __init__(self,file_name=None):
        '''Parameterized constructor of matrix class Takes input: Filename'''
        #intializing class variables
        self.array_2d = None
        if file_name != None:
            self.load_from_csv(file_name)
    def load_from_csv(self,file_name):
        ''' Method to read a file and extract array from it using numpy. input: filename '''
        try:
            self.array_2d = np.genfromtxt(file_name, delimiter=',',dtype = float)
        except:
            print('Exception occured while loading a file')
    def standardise(self):
        '''Method to standardise the array_2d'''
        try:
            #Getting the max,min and mean from the given array
            mean = np.mean(self.array_2d, 0)      
            maximum = np.max(self.array_2d, 0)
            minimum = np.min(self.array_2d, 0)
            #subtracting column mean from every element in the array with 
            self.array_2d=(self.array_2d-mean)/(maximum-minimum)
        except:
            print('Exception occured while standardising the data.')
    def get_distance(self,other_matrix,weights,beta):
        '''method to calculate distance from the row to the centroids inuputs: other_matrix, weights,beta output: Distances'''
        try:
            cents = other_matrix.centroids.shape[0]
            self.beta = beta
            self.matrix_row = self.matrix_row.reshape(1,-1)
            weights = weights.weights
            if self.matrix_row.shape[0] == 1:
                distance = np.zeros((cents, 1))
                # for every row in other_matrix
                for i in range(cents):
                    distance[i, :] = np.sum(np.dot(np.power(weights, beta).reshape(-1, 1), 
                                        np.square(self.matrix_row.reshape(1, -1) - other_matrix.centroids[i, :].reshape(1, -1))))
                return distance
        except:
            print('Exception occured while calculating the distance')

    def get_count_frequency(self):
        '''Method to reutrn count of frequency of clusters from S matrix'''
        result_dict = {}
        #checking the number of columns
        # it'll only work if the column size is 1
        if self.S.shape[1]==1:
            for i in self.S:
                if i[0] in result_dict:
                    result_dict[i[0]]+=1
                else:
                    result_dict[i[0]]=1
            return result_dict
        else:
            return 'column size more than 1'
# functions
    
def get_initial_weights(m):
    '''function to generate initial weights'''
#     generating m random numbers 
#     np.random.seed(42) 
    try:
        weight = np.random.rand(1,m)
        weights = weight/np.sum(weight)
        return weights[0]
    except:
        print('Exception occured while intializing weights')

def get_centroids(m,S,K):
    '''Function takes 3 parameters matrix m, matrix S, K and returns the optimal centroids'''
    try:
        data = m.array_2d
        #clusters should be greater than 2 and lessthan n-1
        if K>=2 and K<len(data.data):
            #checking the array s has any values other than zeroes
            if S.S.any():
                #creating an array to hold centroids
                result_array = np.zeros((K,len(data[0])),dtype = float)
                for i in range(1,K+1):
                    #calculating centroids based on the mean
                    # centroids starts from 1 to k.
                    np.seterr(all="ignore")
                    if len(S.get_count_frequency())>1:
                        j = np.mean(data[S.S.reshape(len(S.S),)==i],axis = 0)
                    #0th posistion of result_array is 1 st centroid.
                        result_array[i-1]=j
                m.centroids = result_array
                return m
            else:
                #When S contains only 0's
                #this means that the method is called for the first time and values of S are initialized to 0's
                numbers = np.random.randint(0, len(data), K)
                m.centroids = data[numbers]
                return m

        else:
            return -1
    except:
        print('Exception occured in Get_centroids Function')
#This function returns matrix S
def get_groups(m,K,beta):
    '''Function takes 3 parameters matrix m, no of clusters to be made K and Beta. This function return the matrix S'''
#     as the object of matrix is passed. getting array_2d from the object
    try:
        data = m.array_2d
        m.K = K

        m.S = np.zeros((len(data),1),dtype = int)
        m.weights = get_initial_weights(len(data[0]))
        centroids = get_centroids(m,m,K)
        for i in range(len(data)):
            #calculating distance in the 1st epoch
            m.matrix_row=data[i]

            distance = m.get_distance(centroids,m,beta)
            m.S[i]= np.argmin(distance)+1

        count = 0
# calculating distances after the 1st epoch
        while True:
#             Getting updated matrix S
            S = np.copy(m.S)
#             Getting updated centroids
            centroids = get_centroids(m,m,K)
            get_new_weights(m,m,m)
            for i in range(len(data)):
                m.matrix_row = data[i]
                distance = m.get_distance(centroids,m,beta)
                m.S[i]=np.argmin(distance)+1
            count += 1
            if np.array_equal(S, m.S):
                break
        S = matrix()
        S.S = m.S
        if len(m.get_count_frequency()) == 1:
            return get_groups(m,K,beta)

        return S
    except:
        print('Exception Occured while making groups.')
            
def get_new_weights(m,centroids,S):
    '''Function to calculate weights. input: matrix m, centroids matrix, S matrix. Output: weights matrix '''
    #Calculating dispersion of the column j in data matrix.
    dispersion_j = []
    centroids = centroids.centroids
    S = S.S
    for j in range(len(m.array_2d[0])): 
        list_outer = []
        for k in range(0,len(centroids)):
        #S matrix has values 1,2,....,k
            list_inner=[]
            for i in range(len(m.array_2d)):
                if m.S[i] == k+1:
                    u_ik = 1
                else:
                    u_ik = 0
                delta_i = u_ik*((m.array_2d[i][j]-centroids[k][j])**2)
                list_inner.append(delta_i)
            
            list_outer.append(sum(list_inner))
        dispersion_j.append(sum(list_outer))
    #calculating wj
    weight_j = []
    col_len = len(m.array_2d[0])
    for j in range(col_len):
        if dispersion_j[j]==0:
            weight_j.append(0)
        else:
            #calculating denominator of the formula given in appendix
            values_denom = [pow((dispersion_j[j]/dispersion_j[t]),1/(m.beta - 1)) for t in range(col_len)]
            #calculating weight_j
            weight_j.append(1/sum(values_denom))
    m.weights = weight_j
    return m

def run_test():
    m = matrix('Data.csv')
    m.standardise()
    for k in range(2,5):
        for beta in range(11,25):
            S = get_groups(m, k, beta/10) 
            print(str(k)+'-'+str(beta)+'='+str(S.get_count_frequency()))
run_test()