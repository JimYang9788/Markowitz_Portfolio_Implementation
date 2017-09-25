
# coding: utf-8

# In[1]:


aapl_minute_closes = get_pricing(
    'SNAP',
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2017-03-01', #customize your pricing date range
    end_date = '2014-05-01',
    frequency='minute', #change to daily for daily pricing
)

# matplotlib is installed for easy plotting
aapl_minute_closes.plot()


# In[4]:


for i in range(0, 4):
    print ("This is number %s" % i)


# In[5]:


result = get_backtest('536a6d181a4f090716c383b7')
result.preview('risk')


# In[6]:


# Apple_Rev is a CSV from Quandl showing the quarterly revenue for Apple.
# It is pre-loaded in your data folder.

AAPLRev = local_csv('AAPL_Rev.csv', date_column = 'Date', use_date_column_as_index = True)
AAPLRev.plot()


# In[7]:


db_minute_closes = get_pricing(
    'DB',
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2017-01-01', #customize your pricing date range
    end_date = '2017-07-01',
    frequency='minute', #change to daily for daily pricing
)

# matplotlib is installed for easy plotting
db_minute_closes.plot()


# In[8]:


db_minute_closes = get_pricing(
    'DB',
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2017-01-01', #customize your pricing date range
    end_date = '2017-07-01',
    frequency='minute', #change to daily for daily pricing
)

# matplotlib is installed for easy plotting
db_minute_closes.plot()


# In[9]:


db_minute_closes


# In[10]:


db_minute_closes = get_pricing(
    'DB',
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2017-02-01', #customize your pricing date range
    end_date = '2017-02-26',
    frequency='minute', #change to daily for daily pricing
)

# matplotlib is installed for easy plotting
db_minute_closes


# In[11]:


db_minute_closes = get_pricing(
    'DB',
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2017-02-01', #customize your pricing date range
    end_date = '2017-02-26',
    frequency='daily', #change to daily for daily pricing
)

# matplotlib is installed for easy plotting
db_minute_closes


# In[12]:


db_minute_closes = get_pricing(
    'DB',
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2017-01-01', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)

# matplotlib is installed for easy plotting
db_minute_closes


# In[13]:


db_minute_closes = get_pricing(
    'DB',
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)

# matplotlib is installed for easy plotting
db_minute_closes


# In[14]:


db_minute_closes = get_pricing(
    'DB',
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)

# matplotlib is installed for easy plotting
for i in db_minute_closes:
    print i[3]


# In[2]:


db_minute_closes = get_pricing(
    'AAV',
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)

# matplotlib is installed for easy plotting
for i in db_minute_closes:
    print i


# In[16]:


db_minute_closes = get_pricing(
    'DB',
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)

# matplotlib is installed for easy plotting
db_prices = for i in db_minute_closes:
    print i
    
db_prices 


# In[17]:


db_minute_closes = get_pricing(
    'DB',
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)

# matplotlib is installed for easy plotting
db_prices = for i in db_minute_closes:
    print i
    
db_prices 


# In[18]:


db_minute_closes = get_pricing(
    'DB',
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)

# matplotlib is installed for easy plotting
db_price = []
for i in db_minute_closes:
    db_price = db_price.insert (i)
    
db_prices 


# In[19]:


db_minute_closes = get_pricing(
    'DB',
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)

# matplotlib is installed for easy plotting
db_price = []
for i in db_minute_closes:
    db_price.insert (0,i)
    
db_prices 


# In[20]:


db_minute_closes = get_pricing(
    'DB',
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)

# matplotlib is installed for easy plotting
db_price = []
for i in db_minute_closes:
    db_price.insert (0,i)
    
db_price


# In[21]:


db_minute_closes = get_pricing(
    'DB',
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)

# matplotlib is installed for easy plotting
db_price = []
for i in db_minute_closes:
    db_price.insert (0,i)
    
db_price


# In[22]:


db_minute_closes = get_pricing(
    'DB',
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)

# matplotlib is installed for easy plotting
db_price = []
for i in db_minute_closes:
    db_price.insert (0,i)
    
db_price

import numpy as np
np.mean (db_price)


# In[23]:


appl_minute_closes = get_pricing(
    'APPL',
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)

# matplotlib is installed for easy plotting
appl_price = []
for i in appl_minute_closes:
    appl_price.insert (0,i)
    
import numpy as np
np.mean (appl_price)


# In[24]:


appl_minute_closes = get_pricing(
    'APPL',
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)

# matplotlib is installed for easy plotting
appl_price = []
for i in appl_minute_closes:
    appl_price.insert (0,i)
    
import numpy as np
np.mean (appl_price)


# In[25]:


appl_minute_closes = get_pricing(
    'AAPL',
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)

# matplotlib is installed for easy plotting
appl_price = []
for i in appl_minute_closes:
    appl_price.insert (0,i)
    
import numpy as np
np.mean (appl_price)


# In[26]:


appl_price


# In[27]:


np.cov(db_price, appl_price)


# In[28]:


import numpy as np
def 3_month_return (S1,S2,S3,S4,S5):
    S1_price = get_pricing(
    str (S1),
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)
S1_hist_price = []
for i in S1_hist_price:
    S1_price.insert (0,i)
    return S1_hist_price


# In[29]:


import numpy as np
def 3_month_return (S1,S2,S3,S4,S5):
    S1_price = get_pricing(
    str (S1),
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)
S1_hist_price = []
for i in S1_hist_price:
    S1_price.insert (0,i)
    return S1_hist_price


# In[30]:


import numpy as np
def my_return(S1,S2,S3,S4,S5):
    S1_price = get_pricing(
    S1,
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)
    S1_hist_price = []
    for i in S1_hist_price:
        S1_price.insert (0,i)
    return S1_hist_price


# In[31]:


my_return (AAPL,AAPL,AAPL,AAPL,AAPL)


# In[32]:


import numpy as np
def my_return(S1,S2,S3,S4,S5):
    S1_price = get_pricing(
    S1,
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)
    S1_hist_price = []
    for i in S1_hist_price:
        S1_price.insert (0,i)
    return S1_hist_price


# In[33]:


my_return ("AAPL","AAPL","AAPL","AAPL","AAPL")


# In[34]:


import numpy as np
def my_return(S1,S2,S3,S4,S5):
    S1_price = get_pricing(
    S1,
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)
    S1_hist_price = []
    for i in S1_price:
        S1_hist_price.insert (0,i)
    return S1_hist_price


# In[35]:


my_return ("AAPL","AAPL","AAPL","AAPL","AAPL")


# In[36]:


import numpy as np
def my_return(S1,S2,S3,S4,S5):
    S1_price = get_pricing(
    S1,
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)
    S1_hist_price = []
    for i in S1_price:
        S1_hist_price.insert (0,i)
        
    S2_price = get_pricing(
    S2,
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)
    S2_hist_price = []
    for i in S2_price:
        S2_hist_price.insert (0,i)
        
    S3_price = get_pricing(
    S3,
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)
    S3_hist_price = []
    for i in S3_price:
        S3_hist_price.insert (0,i) 

    S4_price = get_pricing(
    S4,
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)
    S4_hist_price = []
    for i in S4_price:
        S4_hist_price.insert (0,i) 
 
    S5_price = get_pricing(
    S5,
    fields='close_price', #modify to price, open_price, high, low or volume to change the field
    start_date='2016-11-20', #customize your pricing date range
    end_date = '2017-02-20',
    frequency='daily', #change to daily for daily pricing
)
    S5_hist_price = []
    for i in S5_price:
        S5_hist_price.insert (0,i) 
        
    return [S1_hist_price,S2_hist_price,S3_hist_price,S4_hist_price,S5_hist_price]


# In[37]:


my_return("AAPL","DB","JPM","YHOO","BAC")


# In[38]:


import numpy as np
def my_return(los):
    if los == []:
        return []
    else:
        stock_price = get_pricing(
        los[0],
        fields='close_price', #modify to price, open_price, high, low or volume to change the field
        start_date='2016-11-20', #customize your pricing date range
        end_date = '2017-02-20',
        frequency='daily', #change to daily for daily pricing
)
        hist_price = []
        for i in stock_price:
            hist_price.insert (0,i)
        return hist_price + my_return (los[1:])

        


# In[39]:


my_return (["AAPL"])


# In[40]:


my_return (["AAPL","DB"])


# In[41]:


import numpy as np
def my_return(los):
    if los == []:
        return []
    else:
        stock_price = get_pricing(
        los[0],
        fields='close_price', #modify to price, open_price, high, low or volume to change the field
        start_date='2016-11-20', #customize your pricing date range
        end_date = '2017-02-20',
        frequency='daily', #change to daily for daily pricing
)
        hist_price = []
        for i in stock_price:
            hist_price.insert (0,i)
        return [hist_price] + my_return (los[1:])
    
my_return (["AAPL","DB"])


# In[42]:


import numpy as np
def my_return(los):
    if los == []:
        return []
    else:
        stock_price = get_pricing(
        los[0],
        fields='close_price', #modify to price, open_price, high, low or volume to change the field
        start_date='2016-11-20', #customize your pricing date range
        end_date = '2017-02-20',
        frequency='daily', #change to daily for daily pricing
)
        hist_price = []
        for i in stock_price:
            hist_price.insert (0,i)
        return [hist_price] + my_return (los[1:])
    
my_return (["AAPL","DB"])


# In[43]:


def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks


# In[44]:


optimal_portfolio(my_return (["AAPL","DB"]))


# In[45]:


import cvxopt as opt


# In[46]:


optimal_portfolio(my_return (["AAPL","DB"]))


# In[47]:


get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd


# In[48]:


import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd


# In[49]:


def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks


# In[50]:


optimal_portfolio(my_return (["AAPL","DB"]))


# In[51]:


weights, returns, risks = optimal_portfolio(my_return (["AAPL","DB"]))

plt.plot(stds, means, 'o')
plt.ylabel('mean')
plt.xlabel('std')
plt.plot(risks, returns, 'y-o')


# In[ ]:


def optimal_portfolio(returns, risk_level):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    
    risks_copy = risks
    
    if risks_copy 
    return 


# In[ ]:


my_return (["BLUE","AAPL","DB"])


# In[ ]:


my_return (["BLUE","AAPL"])


# In[52]:


my_return (["DB","AAPL"])


# In[53]:


my_return (["BLUE","AAPL"])


# In[54]:


def optimal_portfolio(returns, risk_level):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    
    risks_copy = risks

    return 


# In[55]:


def get_index (lon,r)
    if lon ==[]:
        return []
    elif lon[0] <= r:
        return [lon.index(lon[0])] + get_index(lon[1:],r)
    else:
        return get_index (lon[1:],r)


# In[56]:


def get_index (lon,r):
    if lon ==[]:
        return []
    elif lon[0] <= r:
        return [lon.index(lon[0])] + get_index(lon[1:],r)
    else:
        return get_index (lon[1:],r)


# In[57]:


get_index ([4,3,4,2,6],3)


# In[58]:


def get_index_2 (lon,r):
    loc = lon
    if lon ==[]:
        return []
    elif lon[0] <= r:
        return [loc.index(lon[0])] + get_index(lon[1:],r)
    else:
        return get_index (lon[1:],r)


# In[59]:


get_index ([4,3,4,2,6],3)


# In[60]:


get_index ([4,3,4,2,6],3)


# In[61]:


[4,3,2,1].index([2])


# In[62]:


[4,3,2,1].index(2)


# In[63]:


def get_index_2 (lon,r):
    loc = lon
    if lon ==[]:
        return []
    elif lon[0] <= r:
        return [loc.index(lon[0])] + get_index_2(lon[1:],r)
    else:
        return get_index_2 (lon[1:],r)


# In[64]:


get_index_2 ([4,3,2,1],3)


# In[65]:


def get_index_2 (lon,r):
    loc = lon[]
    if lon ==[]:
        return []
    elif lon[0] <= r:
        return [loc.index(lon[0])] + get_index_2(lon[1:],r)
    else:
        return get_index_2 (lon[1:],r)


# In[66]:


def get_index_2 (lon,r):
    loc = lon[:]
    if lon ==[]:
        return []
    elif lon[0] <= r:
        return [loc.index(lon[0])] + get_index_2(lon[1:],r)
    else:
        return get_index_2 (lon[1:],r)
    return loc


# In[67]:


get_index_2 ([4,3,2,1],3)


# In[68]:


def get_index_2 (lon,r):
    loc = lon[:]
    if lon ==[]:
        return []
    elif lon[0] <= r:
        return [loc.index(lon[0])] + get_index_2(lon[1:],r)
    else:
        return get_index_2 (lon[1:],r)
    return loc


# In[69]:


get_index_2 ([4,3,2,1],3)


# In[70]:


def get_index_2 (lon,r):
    loc = lon[:]
    if lon ==[]:
        print loc
        return []
    elif lon[0] <= r:
        return [loc.index(lon[0])] + get_index_2(lon[1:],r)
    else:
        return get_index_2 (lon[1:],r)


# In[71]:


get_index_2 ([4,3,2,1],3)


# In[72]:


def get_index_2 (lon,r):
    loc = lon
    return get_index (lon, r)
   
def get_index (lon, r)    
    if lon == []:
        return []
    elif lon[0] <= r:
        return [loc.index(lon[0])] + get_index_2(lon[1:],r)
    else:
        return get_index_2 (lon[1:],r)


# In[73]:


def get_index_2 (lon,r):
    loc = lon
    return get_index (lon, r)
   
def get_index (lon, r):    
    if lon == []:
        return []
    elif lon[0] <= r:
        return [loc.index(lon[0])] + get_index_2(lon[1:],r)
    else:
        return get_index_2 (lon[1:],r)


# In[74]:


get_index_2 ([4,3,2,1],3)


# In[75]:


def get_index_2 (lon,r):
    loc = lon
    return get_index (lon, r, loc)
   
def get_index (lon, r, loc):    
    if lon == []:
        return []
    elif lon[0] <= r:
        return [loc.index(lon[0])] + get_index(lon[1:],r,loc)
    else:
        return get_index (lon[1:],r,loc)


# In[76]:


get_index_2 ([4,3,2,1],3)


# In[77]:


def get_index_2 (lon,r):
    loc = lon
    return get_index (lon, r, loc)
   
def get_index (lon, r, loc):    
    if lon == []:
        return []
    elif lon[0] <= r:
        return [loc.index(lon[0])] + get_index(lon[1:],r,loc)
    else:
        return get_index (lon[1:],r,loc)


# In[78]:


get_index_2 ([4,3,2,1],3)


# In[79]:


def get_stuff (target, indices):
    if indices == []:
        return []
    else:
        return [target[indices[0]]] + get_stuff (target , indices[1:])
get_stuff ([7,5,7,4,1],[0,2,3])


# In[80]:


def optimal_portfolio(returns, risk_level):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    
    list_of_stock = get_stuff (returns ,get_index_2 (risks, risk_level))

    return list_of_stock

optimal_portfolio (my_return (["DB","AAPL"],3))


# In[81]:


def optimal_portfolio(returns, risk_level):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    
    list_of_stock = get_stuff (returns ,get_index_2 (risks, risk_level))

    return list_of_stock

optimal_portfolio (my_return (["DB","AAPL"]),3)


# In[82]:



optimal_portfolio (my_return (["DB","AAPL","BAC","YHOO","BLUE"]),3)


# In[83]:


my_return (["DB","AAPL","BAC","YHOO","BLUE"])


# In[84]:


optimal_portfolio (my_return (["DB","AAPL","BAC","YHOO"]),3)


# In[85]:


optimal_portfolio (my_return (["DB","AAPL","YHOO"]),3)


# In[86]:


my_return (["TECK.B"])


# In[6]:


def call_price (ticker, exp, strike_price):
    
    closing_prices = get_pricing(
        ticker,
        fields='close_price', #modify to price, open_price, high, low or volume to change the field
        start_date='2016-17-20', #customize your pricing date range
        end_date = '2017-07-20',
        frequency='daily', #change to daily for daily pricing
    )

for i in closing_prices:
    print i
    

