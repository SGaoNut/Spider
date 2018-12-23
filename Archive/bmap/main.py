sen = "abc,123,4,789,mnp"
replace_word = ['abc','mnp']
sen = sen.replace(replace_word, "gaoshan")

string = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+', "",line)