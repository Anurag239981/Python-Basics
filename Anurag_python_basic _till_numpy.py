#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###Basic of anaconda:---

3 options:
    anaconda navigator
    anaconda prompt - jupyter notebook
    jupyter notebook
    
Jupyter notebook:
    cell
    
properties
    3 modes: RawNBConvert (Esc + R) | Markdown (Esc + M) | Code (Esc + Y)
    2 states: Select (Esc) | Edit (Click in the cell)
        
    other shortcuts:
    cut (Esc + X)
    copy (Esc + C)
    paste (Esc + V)
    
    adding the cell above (Esc + A)
    
    adding a cell below (Esc + B)
    
    delete an existing cell (Esc + DD)
    
    Execute the contents in the cell:
        Shift + Enter
        Cntr + Enter


# ### Functions

# In[ ]:


functions can be of two types
    1. inbuilt functions
    2. UDFs - user defined functions

type() - to get the datatype / type of a variable / object

print() - to print the content of the variable / object
        this will not return the output; will only print the value
        we can't use this output for further calculations


# ### Properties of Python

# In[ ]:


easy language to learn

dynamically typed / loosely typed

in python, any text that we want to write will be in either of single or double quotes. 
    We may also use three single / double quotes for the multiline text

python is case sensitive

in basic python, most of the keywords will be in small case.
    exceptions: True, False, None
        
commnets
    the line if the codes/text that will not execute as a part of a code
    using # symbol
    python supports ONLY single line commnets


# #### INC (Identifier Naming Conventions) rules

# In[ ]:


names can be alphanumeric, mix of alphabets and numbers

no special chars allowed except underscore

names can start with an alphabet or underscore
    **dont start the names with underscore

**no inbuilt keyword or function names

no limit on the length of the chars in the names
    **use short but meaningful names


# #### variables

# In[2]:


var=5


# In[3]:


type(var)


# In[5]:


var=10.5


# In[6]:


type(var)


# In[7]:


var


# In[8]:


var+10


# In[9]:


print(var) + 10             ##error reason:-  print() - to print the content of the variable / object
                                                     ##this will not return the output; will only print the value
                                                     ##we can't use this output for further calculations


# #### data types

# for the better memory allocations; generally we will have many datatypes when we work on tools which can hanle the data
# 
# int, smallint, long, tinyint, bigint
# 
# double, float, numeric, single
# 
# str, text, varchar, char, nchar, nvarchar
# 
# boolean, binary
# 
# date, datetime, time
# 
# complex
# 
# blob
# 
# Basic python supports only 4 of them:
# str
# float
# int
# bool
# 
# Data type conversion functions
# str()
# float()
# int()
# bool()
# 
# ---Non standard numeric: Numeric data that is stored as str
# 
# --text cant be converted into numbers; only we can convert non standard numerics into numerics
# 
# python doesnt supports the implicit data type conversion

# In[ ]:


var1 = 10                   ##INC(identifier naming conventions)
var2 = 15.5
var_3 = 'Python'
_v1 = True

# 1var = 10 - incorrect


# ### operators

# In[ ]:


symbols which can be used to perform the calculations

assignment: =

arithmatic: + - * / // ** %

arithmatic + assignment: += -= *= /= ...

comparision | relational | conditional - output will always be boolean (True or False )

single valued: > >= < <= != <> ==

multi valued | membership : in, not in
logical and or not

and / or : combining the boolean values

    * and: Output will be True only if all the booleans are True

    * or: Output will be True if any one of booleans is also True

    * not: negation

bitwise 
    **WE WILL DISCUSS THIS IN CLASS 4 OR 5
    & | ~
Sequence of operations:

B () --> O [**] --> D M [/ // % *] --> S A [+ -] --> Conditional [> >= < <= <> == in not in] --> not --> and --> or


# In[ ]:


# Q1: What is the output of following expression
5 + 4 * 9 % (3 + 1) / 6 - 1 


# 5 + 4 * 9 % (3 + 1) / 6 - 1
# 
# 5 + 4 * 9 % 4 / 6 - 1
# 
# 5 + 36 % 4 / 6 - 1
# 
# 5 + 0 / 6 - 1
# 
# 5 + 0.0 - 1
# 
# 5.0 - 1
# 
# 4.0

# ### control flow statements

# In[ ]:


1. Conditional statements
        
    if elif else
    
    syntax:
    -------------------------------------
    
    if condition/s:
        line1
        line2...
        
    elif condition/s:
        line3
        line4..
        
    ....
    
    else:
        line n..
    
    
    ternary if : single line statement for condition checking 
    
    Excel:
    if( condition/s, True_value, False_value )
    
    Syntax:
    True_value if condition/s else False_value
    

2. Iterative statements
    for
    while    


# In[13]:


# display if the given number is positive or negative
num = 0

if num > 0:
    print('positive')
else:
    print('negative')


# In[14]:


print( 'positive' if num > 0 else 'negative' )   ##ternary if


# In[15]:


num = -10

if num > 0:
    print('positive')
else:
    print('negative')


# In[16]:


# display if the given number is positive or negative or zero
num = 0

if num > 0:
    print('positive')
    
elif num == 0:
    print('zero')
    
else:
    print('negative')


# In[17]:


num = -10
'positive' if num > 0 else 'zero' if num == 0 else 'negative'


# In[18]:


num = 10

if num > 0:
    print('positive')


# ####ENDCLASS1

# In[ ]:


To get the documentation of python:

    1. use ? after the keyword or function_name and execute the cell
    2. Click within the brakets of the function and press SHIFT  + TAB


# #### Loops

# In[ ]:


**while**

Properies:
    * conditional loop
    * entry controlled loop
    * will execute for true conditions
    * must have atleast one statement to make the conditions False
    
Syntax:
    
while conditions: 
    line1
    line2... 
    statement to make the conditions False
    

**for**

properties:
    * ranged loop
    * we would know how many time the loop will execute
   
syntax:

for var_name in iterable: 
    line1 
    line2...
    

**iterable**
an object that contains multiple values


**range()**
function that generates range of integer numbers

syntax:

    range( start, end, step )
    range( start, end )          # step = 1
    range( end )                 # start = 0, step = 1
    
    **range will generate the numbers from**
    
        1. start till end - 1, if step is positive
        2. start till end + 1, if step is negative
        
    we can't generate the floating numbers with range() function


# In[19]:


num = 11
while num <= 10:
    print(num)


# In[1]:


num = 1
while num <= 10:
    print(num)
    num = num + 1


# In[ ]:


range(1, 5, 1)    # 1, 2, 3, 4

range( 5 )        # 0, 1, 2, 3, 4

range(1, 10, 2)   # 1, 3, 5, 7, 9


# In[2]:


range(1, 5, 1)


# In[3]:


for v1 in range( 1, 5, 1):    # 1, 2, 3, 4
    print(v1)


# In[4]:


for v1 in range( 5 ):
    print(v1)


# In[5]:


for v1 in range( -2, 5 ):
    print(v1)


# In[6]:


for v1 in range( 2, 11, 2 ):
    print(v1)


# In[7]:


for v1 in range( 5, 1, -1 ):
    print(v1)


# In[8]:


for v1 in range( -9, -1 ):
    print(v1)


# #### data structures

# In[ ]:


data structures will give us a way to store the data

basic python supports only 4 data structures:
tuple, list, dict, set

dict - key and value pairs
set - keyless dict

immutable: object that CANT be updated in the same memory location
mutable: object that CAN be updated in the same memory location; we will use the mutable object often in the data analysis

properties of data structues:
    1. all data structures are 1 Dimensional
    2. all of them are hetrogeneous  ex:- t1=(1,name,-2)
    3. None of them will allow broadcasting ex:- one in front of multiple values
    4. Allow vectorization ex:- one to one mapping
    
1. How to create

2. How to access the data

3. How to modify or update the data 

4. What are the Methods ( functions ) and Attributes ( variables ) of data structures

5. How to perform the mathematical calculations

6. How to perform the indexing - condition checking

            create               access     
----------------------------------------------------------------------------------
tuple       no brackets          []
            ()
            tuple()

list        []                   []
            list()
            

dict        {}                   []



set         {}                   []


type conversion functions - tuple(), list(), dict(), set()

Access the data: []

tuple and list - 
    1. get the value from the position: we can get only one element in op
        ds_name[ pos ]
    
    2. in case we want multiple elements in output, use slicers 
        ds_name[ start : end : step ]


# In[20]:


t1=10


# In[21]:


type(t1)


# In[22]:


t2=10,20,30,40,50


# In[23]:


type(t2)


# In[24]:


t3 = ( 1, 2, 10.5, True, 'Python', t1, t2 )


# In[25]:


type(t3)


# In[26]:


t3


# In[27]:


l1 = [10, 20, 30, 40, 50]


# In[28]:


type(l1)


# In[29]:


l2 = [10, 10.5, True, False, 'Python', t2, l1 ]


# In[30]:


l2


# In[31]:


t5 = tuple( range(10, 51, 10) )


# In[32]:


t5


# In[33]:


l3 = list( range(1, 11, 2) )


# In[34]:


l3


# In[35]:


t2


# In[36]:


t2[0]


# In[37]:


t2[-1]


# In[38]:


l1


# In[39]:


l1[2]


# In[40]:


l1[-4]


# In[41]:


t2


# In[43]:


t2[: :]


# In[44]:


t2[ 1:3:1 ]


# In[45]:


t2[:3]      #means zero on the first position


# In[46]:


t2[2:]       ##means start with index 2 and go till end 


# In[47]:


l1


# In[48]:


l1[::-1]


# In[49]:


l1[-3:-1:1]


# In[50]:


l1[-2:-4:-1]


# In[51]:


l1[1::2]


# In[52]:


l1[2] = -1      ## we can change value in list in same memory location(id) but in tuple we cant change value in same memory location


# In[53]:


l1


# In[54]:


id(l1)


# In[55]:


id(l2)


# ### OOP

# In[ ]:


Object Oriented Programming

customer: name, ph, email, gender, dob
product
inventory
order
transaction
house
survey
payment modes
employee
vendors
transport
.....


class: blueprint or template
    name, ph, email, gender, dob
    functionalities
    
object: when we give the values in the template; that will become object of that class
    
objects:
    methods - functions
    attributes - variable
    
function_name()
variable_name

object_name.method_name()
object_name.attribute_name


# In[56]:


print( dir(tuple) )


# In[57]:


t9 = 1, 2, 3, 4, 3, 2, 1, 4, 5, 6, 5, 4, 3, 2, 1, 4


# In[58]:


t9.count(5)


# In[59]:


t9.index( 5 )


# In[60]:


t9.index( 5, 9 )


# In[61]:


t9.index( 5, t9.index( 5 ) + 1 )


# In[ ]:


###ENDCLASS2


# In[ ]:


# NoSQL - Not Only SQL
SQL - structured data: tabular data in form of rows and columns
Semi Structured - csv, json, xml or any delimted data etc.
unstructured - text, image, video, audio

##Last class topics covered:-
functions: range(), print(), type(), dir(), id(), input(), len()
type conversion functions: int(), float(), bool(), str(), list(), tuple(), set(), dict()

Loops: while, for
    
Data Structures: tuple, list, dict, set
    1D, hetrogeneous, no broadcasting, allow vectorization
    
    tuple - immutable (read only) : cant update the data in SAME MEMORY location
    list - mutable (read write)
    
    * create
    * access the data
        i. get the vlaue from position / index
            obj[ pos ]
        ii. slicers
            obj[ start : end : step ]
        iii. indexing - get the values based on conditions
    * how to update the data; if possible
    * methods and attributes
        obj_name.method_name()
        obj_name.attribute_name
        
in - multi valued operator


# #### attributes and methods of a list

# In[62]:


print( dir( list ))


# In[63]:


l1 = [ 1, 2, 3, 2, 1, 3, 4, 3, 2, 4, 1 ]


# In[64]:


l1.count(1)


# In[65]:


l1.index(4)


# In[66]:


l1.index( 4, l1.index(4) + 1 )


# In[67]:


l1.append(10)


# In[68]:


l1


# In[69]:


l1.append( t1 )


# In[70]:


l1


# In[71]:


len(l1)


# In[72]:


l1.extend([10])


# In[73]:


l1


# In[74]:


l1.insert( 0, -1 )


# In[75]:


l1


# In[76]:


l1.pop()


# In[77]:


l1


# In[80]:


l1.pop( 4 )


# In[81]:


l1


# In[82]:


l1.remove( 3 )                       #only first 3 will be removed 


# In[83]:


l1


# In[84]:


l1.reverse()


# In[85]:


l1


# In[86]:


l1.sort( reverse = False ) 


# In[87]:


l1


# In[90]:


l1=[10,20,30,40,50]


# #### Q0: print the positions of range

# In[91]:


for var in range( 5 ):     # [0, 1, 2, 3, 4]
    print( var )


# ##### Q1: Print all the values in the list l1 in separate lines

# In[92]:


# iterations on the positions
for var in range( len(l1) ):     # [0, 1, 2, 3, 4]
    print( l1[var] )


# In[93]:


# iterations on the values
for var in l1:    # [10, 20, ... 50]
    print(var)


# #### Q2: Print all the values in the list l1 in separate lines after adding 100 to every element

# In[94]:


# iterations on the positions
for var in range( len(l1) ):     # [0, 1, 2, 3, 4]
    print( l1[var] + 100 )


# In[95]:


# iterations on the values
for var in l1:    # [10, 20, ... 50]
    print( var + 100 )


# #### Q3: Update the list l1 with the values in the list l1 after adding 100 to every element

# In[96]:


# iterations on the positions
for var in range( len(l1) ):     # [0, 1, 2, 3, 4]
    l1[var] = l1[var] + 100


# In[97]:


l1


# In[98]:


# iterations on the values
i = 0
for var in l1:    # [10, 20, ... 50]
    l1[i] = var + 100
    i = i + 1


# In[99]:


l1


# #### Q4: Create a new list l2 with the values in the list l1 after adding 100 to every element

# In[100]:


l1 = [10, 20, 30, 40, 50]


# In[101]:


# iterations on the positions
l2=[]
for var in l1:
    l2.append(var+100)


# In[102]:


l2


# In[103]:


# iterations on the positions
l2 = []
for var in range( len(l1) ):     # [0, 1, 2, 3, 4]
    l2.append( l1[var] + 100 )
    
print(l1)
print(l2)


# In[104]:


l1 + 100   # broadcasting is NOT possible


# ### Q5: Create a new list l3 with the values in the list l1 which are GE 30

# In[105]:


l1 >= 30     # broadcasting is NOT possible


# In[106]:


l3 = []
for var in l1:
    if var >= 30:
        l3.append( var )

print(l1)
print(l3)


# ### list comprehensions

# In[ ]:


* is a iterative statement for single line operations

syntax:  
[ var for var in iterable ]

We can use list comprehensions to:
 
 1. mathematical operations
 2. indexing
 
 
mathematical operations: 
[ var + 100 for var in iterable ]

conditional operations / indexing: 
[ var for var in iterable if condition/s ]


# In[107]:


l1


# In[108]:


# mathematical operations:
l2 = [ (i / 100) + 10 for i in l1 ]


# In[109]:


l2


# In[110]:


# indexing
l2 = [ i for i in l1 if i >= 30 ]


# In[111]:


l2


# ### combine the concepts of math ops and indexing

# In[112]:


l1 = [1, 2, 3 , 4, 5, 4, 3, 1, 2, 3, 4, 5 , 6]


# ##### get the even numbers from list l1, and display the output after adding 10 to all the resultant numbers

# In[113]:


# method 1
l2 = []
for x in l1:    # [1, 2, 3 , 4, 5, 4, 3, 1, 2, 3, 4, 5 , 6]
    if x % 2 == 0:
        l2.append( x + 10 )
l2


# In[114]:


# method 2
l2 = [ x + 10 for x in l1 if x % 2 == 0 ]
l2


# ##### get the odd numbers from list l1, and display the output after adding 10 to all the numbers LE 3 and 20 to numbers GT 3

# In[115]:


l1


# In[116]:


[ x for x in l1 if x % 2 != 0 ]


# In[117]:


[ x + 10 if x <= 3 else x + 20 for x in l1 if x % 2 != 0 ]


# In[118]:


l2 = []
for x in l1:
    if x % 2 != 0:
        l2.append( x + 10 if x <= 3 else x + 20 )
l2


# In[119]:


l2 = []
for x in l1:
    if x % 2 != 0:
        
        if x <= 3:
            l2.append( x + 10 )
        else:
            l2.append( x + 20 )
l2


# ### dict

# In[ ]:


key and value pairs

keys can be only of immutable type; however values can be anything
    **guideline: always use keys as int or str type

syntax:
{ key1 : value1, key2 : value2, ....... }


# In[120]:


d2 = {1:10, 2:20, 3:30, 4:40, 5:50}


# In[121]:


type(d2)


# In[122]:


d3 = { 1: 10, 2: 10.5, 3: True, 4: 'Python', 5: t1, 6: l1, 7: d2 }


# In[123]:


d8 = { 'Name' : ['John', 'Andrina', 'Sam'],
         'Salary' : [100, 200, 150],
            'Gender' : ['M', 'F', 'M'] 
     }


# In[124]:


d8


# In[ ]:


###CLASSEND


# In[125]:


# add more than one objecty in the list
l1.extend( [t1, t2] )


# In[ ]:


key and value pairs

keys can be only of immutable type; however values can be anything
    **guideline: always use keys as int or str type

syntax:
{ key1 : value1, key2 : value2, ....... }

We cant access the data out of dict with the key if its not available in the dict

If we have key present in the dict, and we assign a new value; it will update the value of the existing key

If we have key is NOT present in the dict, and we assign a new value; it will assign a new key and  value pair of the dict


# In[126]:


d1 = { 1: 10, 2: 20, 3:30, 4:40, 5:50 }
d2 = { 'Name': 'Sam', 1: 10, 'Salary': 1000, 'Age': 25 }


# In[127]:


d1[4]


# In[128]:


d12 = {'a':'True',True:'False'}     #True work as 1
d12[1]


# In[129]:


print(dir( dict ))


# In[130]:


d2


# In[131]:


d2.keys()


# In[132]:


d2.values()


# In[133]:


d2.items()


# In[134]:


d2.pop( 1 )


# In[135]:


d2.popitem()


# In[136]:


d1


# In[137]:


for i in d1:
    print(i)


# In[139]:


for i in d1.values():
    print(i)


# In[140]:


for i in d1.items():
    print(i)


# ### sets

# In[ ]:


keyless dict


# In[141]:


s1 = { 5, 4, 2, 3, 4, 5, 1, 3, 2, 1 }
s2 = { 1, 2, 6, 7 }


# In[142]:


print(dir(set))


# In[143]:


s1.difference(s2)


# In[144]:


s1.intersection(s2)


# In[145]:


s1.union(s2)


# In[146]:


s1.intersection_update(s2)


# In[147]:


s1


# ### UDFs

# In[ ]:


User Defined Function: Named entity which can be used multiple times once defined in the memory
    **UDFs should have a retrun statement, though it is not mandatory

func_name( a1, a2, /, a3, a4 )

* a1, a2 - can be given ONLy as positional args
* a3, a4 - can be given as positional or keyword args




    1. Function Defination
    --------------------------------
    
    def function_name( a1, a2, ...... ):
        logic...
        return statement
        
    - mandatory arguments
    - optional arguments
    - mix of mandatory and optional : optional argument MUST always follow mandatory arguments 
    
    - *args : n positional arguments, 0 to n and args is a object name
    - **kwargs : n keyword arguments, 0 to n

    2. Function Calling
    ---------------------------------
    -
    function_name( a1, a2, ..... )
    
    - positional arguments
    - keyword arguments
    - mix of positional and keyword: keyword argument MUST always follow positional arguments 
    - / forward slash : is given in ceratin functions / methods to inform all args before / can be used only as positional args
    
    


# ##### create a UDF to sum two numbers

# In[148]:


def sum_2_no_1( n1, n2 ):
    return n1 + n2


# In[149]:


a = sum_2_no_1( 1, 3 )


# In[150]:


sum_2_no_1( 3, 3 )


# In[151]:


def sum_2_no_2( n1, n2 ):
    print( n1 + n2 )


# In[152]:


b = sum_2_no_2( 1, 3 )


# ##### UDF to subtract two numbers

# In[153]:


def sub_numbers( n1, n2 ):
    return n1 - n2


# In[154]:


sub_numbers( 10, 20 )


# In[155]:


sub_numbers( n1 = 20, n2 = 10 )


# In[156]:


sub_numbers( 20, n2 = 10 )


# In[157]:


sub_numbers( n1 = 20, 10 )


# In[158]:


def sub_numbers( n1 = 0, n2 = 0 ):
    return n1 - n2


# In[159]:


sub_numbers()


# In[160]:


sub_numbers( n1 = 10 )


# In[161]:


def sub_numbers( n1, n2 = 0 ):
    return n1 - n2


# In[162]:


def sub_numbers( n1 = 0, n2 ):
    return n1 - n2


# In[163]:


def sub_numbers( n2, n1 = 0 ):
    return n1 - n2


# ##### write a UDF to sum n numbers

# In[1]:


def sum_no(*args):
    return sum(args)
  


# In[2]:


sum_no(1,2,3,5)


# In[3]:


sum_no( 1, 2, 3, 4, 'py' )


# ##### write a UDF to sum n numbers; UDF should take care of mix of datatypes a s well

# In[5]:


def sum_numbers( *args ):
    
    Sum = 0
    for var in args:
        if type(var) == int or type(var) == float:     # if type(var) in (int, float):
            Sum = Sum + var
            
    return Sum


# In[6]:


sum_numbers( 1, 2, 3, 4, 5 )


# ##### write a program to save only int and float value in a list

# In[8]:


t1 = 1, 2, 3, 4, True, None, 'Python', 5


# In[10]:


l1 = []
for var in t1:
    if type(var) in (int, float):
        l1.append( var )
l1


# In[11]:


def sum_numbers( *args ):
    
    l1 = []
    for var in args:
        if type(var) == int or type(var) == float:     # if type(var) in (int, float):
            l1.append( var )
            
    return sum(l1)


# In[12]:


sum_numbers( 1, 2, 3, 4, 5 )


# In[13]:


sum_numbers( 1, 2, 3, 4, True, None, 'Python', 5 )


# In[14]:


def func_name( **kwargs ):
    
    Sum = 0
    for i in kwargs.values():
        if type(i) == int:
            Sum += i    # Sum = Sum + i
    return Sum


# In[15]:


func_name( a = 10, b = 20, c = 30 )


# In[16]:


def func_name( a, b, c, *args, **kwargs ):
    return (a, b, c, args, d, e, kwargs )


# In[ ]:


func_name( 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 20, z = 30, y = 20 )


# In[ ]:


###ENDCLASS


# #### lambda functions

# In[ ]:


Properties:

functions for the single line operations
lambda functions will not have any name and will not be stored in the memeory
lambda functions are NEVER used alone, they are always used along with other functions or methods

in basic python we will use lambda functions with following functions:
    1. map() - mathematical operations
    2. filter() - indexing

syntax:
lambda x: x + 10


# In[17]:


def func( x ):
    return x + 10


# In[18]:


lambda x : x + 10


# In[19]:


func(10)


# In[ ]:


l1 = [1, 2, 3, 4, 5]
t1 = tuple(l1)


# In[21]:


list( map( lambda x: x + 10, l1 ) )    # [1, 2, 3, 4, 5]


# In[22]:


list( map( lambda x: func(x), t1 ) )


# In[23]:


list( map( func, l1 ) )


# In[25]:


t2=1,2,3,4,5


# In[26]:


list( map( lambda x, y: x + y , l1, t2 ) )


# #### install the packeges

# In[ ]:


pip install package_name

    * package was not available in the system, it will download the latest version of the package in 
      the working environment
    
    * the latest version of the package is available in the working environment; message will come 
      "requirement already satisfied"
      
    * package is available, but new version is available, will ask you new version is available; do 
      you want to install latest version or not

conda install package_name


# In[ ]:


pip install numpy


# ### numpy - numerical python

# In[ ]:


basic python data structures: 4 data structures, hetrogeneous, 1D, broadcasting was not possible, 
                              allow vectorization, datatypes: 4 datatypes, memory consumption was high

numpy

datastructure: ndarray - ND, homogeneous, Allow broadcasting and vectorization


# In[ ]:


0. Methods and Attributes
1. Create

    from type conversion: np.array()
    inbuilt methods: 
        np.zeros(), np.ones(), np.full(), np.identity()
        np.arange(), np.linspace(), np.random.random(), np.random.randint()
        transformations (ndarray): .transpose(), .T, .reshape()
    from the external data: images

2. Access
3. Update
4. Mathematical Operations
5. Indexing

attributes of ndarrays:
.nbytes
.dtype
.T
.shape
.ndim
.size


# In[27]:


import numpy as np


# In[28]:


print(dir( np ))


# In[29]:


l1 = [1, 2, 3, 4, 5]


# In[30]:


a1=np.array(l1)


# In[31]:


a1


# In[32]:


a3 = np.zeros(10)


# In[33]:


a3


# In[34]:


a3.ndim


# In[35]:


a3.shape


# In[36]:


a3.size


# In[37]:


a4 = np.zeros((5, 3))
a4


# In[38]:


a4.ndim


# In[39]:


a4.shape


# In[40]:


a4.size


# In[41]:


a4.T


# In[42]:


a5 = np.ones( (3, 3), dtype = np.int8 )
a5


# In[43]:


a6 = np.full( (3, 3), 'Python', dtype = np.object )
a6


# In[ ]:


###ENDCLASS


# In[ ]:


np.array( object, dtype = )

np.zeros( shape = (), dtype = )
np.ones( shape = (), dtype = )
np.full( shape = (), fill_value = , dtype = )

np.arange() - to generate the range of numbers from start till end -/+ step
np.random.random()
np.random.randint()
np.linspace()

.transpose()
.reshape()

Attributes:
.nbytes
.T
.shape
.size
.ndim


# In[44]:


np.array( range( 1, 21, 1 ) )


# In[45]:


np.arange( 1, 21, 1 )


# In[46]:


np.arange( 0, 1, 0.01 )


# In[47]:


np.linspace( 1, 10, 30 )


# In[48]:


np.random.random( 10 )  # will generate the random number between 0 and 1


# In[50]:


np.random.randint( 10, 100, 20 )


# In[51]:


# generate a 2d array of shape 4 * 5 with 2 digit random numbers
a1 = np.random.randint( 10, 100, 20 )


# In[52]:


a1


# In[53]:


a2 = a1.reshape( (4, 5), order = 'C' )


# In[54]:


a2


# In[55]:


a1.reshape( (4, 5), order = 'F' )


# In[56]:


a3 = a1.reshape((2, 2, 5))


# In[57]:


a3


# In[58]:


import numpy as np
a4 = np.full( (2, 4, 4, 5), 1 )


# In[59]:


a4


# #### Access the data out of numpy array

# In[ ]:


a1[ pos ]
a1[ start : end : step ]


# In[61]:


a1


# In[62]:


a1[3]


# In[63]:


a1[-1]


# In[64]:


a1[ 1:5 ]


# In[65]:


a1[::2]


# In[66]:


a1[1::2]


# In[ ]:


a2[ start : end : step , start : end : step ]
a2[ pos, pos ]
a2[ pos, : ]   # a2[ pos ]
a2[ :, pos ]   # a2[, pos] not possible


# In[67]:


a2


# In[68]:


a2[0, ]


# In[69]:


a2[0, :: ]


# In[70]:


a2[3, :]


# In[71]:


a2[3:, :]


# In[72]:


a2[ 1:3, 1:4 ]


# In[73]:


a2[ 1:, 3: ]


# In[74]:


a2[:, :2]


# In[75]:


a2[1, 1] = -1


# In[76]:


a2


# In[77]:


a2[ 1:3, 2:4 ] = 0    # broadcasting is possible


# In[78]:


a2


# In[79]:


a2[ 1:3, 2:4 ] = np.array( [[1, 2], [3, 4]] )   # vectorization


# In[80]:


a2


# In[81]:


# get the data from a2 where the values are GE 50
a2[ a2 >= 50 ]


# In[82]:


a2


# In[83]:


sum(a2)


# In[84]:


# get the data from a2 where values are more than average
a2[ a2 > a2.mean() ]


# In[ ]:




