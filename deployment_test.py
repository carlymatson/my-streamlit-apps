import streamlit as st

st.write("I have been deployed!")

my_num = st.slider("Input number", min_value = 0, max_value = 10, value = 5)
st.write("Your number is: ", my_num)

my_dict = {'a': 5, 'b': 6}


def plus_a_b(a, b=1):
	return a+b

print(my_dict)
print(plus_a_b(**my_dict))