import re
def getSymbols():
    file_name = "downloadpage001.txt"
    with open(file_name, "r") as f:
        htmlURL = f.read()
    empty = re.findall(r"Sorry, there's nothing here now.", htmlURL)
    if len(empty) > 0:
        return []
    links = re.findall(r'nasdaq-ndx-index__row.*?</tr>', htmlURL)
    finalLinks=[]
    cpnamelist = []
    comList = dict()
    for itemLink in links:
        eachlink = re.findall(r'<a href="/market-activity/stocks/.*?</a>', itemLink)
        symbol = re.findall(r'">.*?</',eachlink[0])[0][2:-2]
        if not (itemLink in finalLinks) : 
            finalLinks.append(symbol)
            cpname = re.findall(r'">.*?</a>',eachlink[1])[0][2:-4]
            cpnamelist.append(cpname)
            comList[symbol] = cpname
    return comList,finalLinks,cpnamelist