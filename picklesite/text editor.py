with open('text.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]
print(content)
for i in range(len(content)):
    content[i] = content[i][1:]
    content[i] = content[i][:50]
    content[i] = content[i][:29:] + content[i][32:]
    print(content[i])


m30, d30 = open_trait.day_gradient(60, 30)
m7, d7 = open_trait.day_gradient(37, 30)
m60_30, d60_30 = open_trait.day_gradient(90, 60)
m14_7, d14_7 = open_trait.day_gradient(44, 37)
m21_14, d21_14 = open_trait.day_gradient(51, 44)
m28_21, d28_21 = open_trait.day_gradient(68, 51)
sumd30 = open_trait.sos_error(d30)
sumd7 = open_trait.sos_error(d7)
sumd60_30 = open_trait.sos_error(d60_30)
sumd14_7 = open_trait.sos_error(d14_7)
sumd21_14 = open_trait.sos_error(d21_14)
sumd28_21 = open_trait.sos_error(d28_21)
pcp = open_trait.previous_close_price("Close")
php = open_trait.previous_close_price("High")
plp = open_trait.previous_close_price("Low")
pop = open_trait.previous_close_price("Open")
rcp_2 = open_trait.recent_close_price(2)
rcp_3 = open_trait.recent_close_price(3)
rcp_4 = open_trait.recent_close_price(4)
rcp_7 = open_trait.recent_close_price(7)
rcp_14 = open_trait.recent_close_price(14)
rcp_17 = open_trait.recent_close_price(17)
rcp_21 = open_trait.recent_close_price(21)
rcp_41 = open_trait.recent_close_price(41)
rcp_61 = open_trait.recent_close_price(61)
rcp_81 = open_trait.recent_close_price(81)
X_open = [m30, sumd30, m7, sumd7, m60_30, sumd60_30, m14_7, sumd14_7, m21_14, sumd21_14, m28_21,
           sumd28_21, pcp, php, plp, pop, rcp_2, rcp_3, rcp_4, rcp_7, rcp_14, rcp_17, rcp_21, rcp_41, rcp_61, rcp_81]
nn_open_output = nn_open.think(X_open)