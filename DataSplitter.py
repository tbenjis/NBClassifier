#
#   This file splits the main PlayStore data files to two,
#   The TopDeveloper Version and NonTop Developer version,
#   It also selects the useful columns and writes to a new file
#   and removes non alphabetic characters.
#

import csv
import re
dataTop, top_row, dataNonTop, nontop_row  = [], [], [], []  # Buffer list

# initial header for new csv file
top_row = ["Category", "AppName", "ContentRating", "IsFree", "HaveInAppPurchases", "Description"]
nontop_row = ["Category", "AppName", "ContentRating", "IsFree", "HaveInAppPurchases", "Description"]

# append the headers
dataTop.append(top_row)
dataNonTop.append(nontop_row)

# open the original file and loop through each line
with open("data/PlayStoreData.csv", "rU") as the_file:
    reader = csv.reader(the_file, quotechar='"', delimiter=";") # delimiter is a semicolon(;)

    for row in reader:

        try:
            #cleanup row data
            desc1 = re.sub(r'[^a-zA-Z ]+','', row[25]) # strip non alphabetical and space chars
            desc = re.sub(r'(?<=(\s\w{15}))(\w*\s)',' ', desc1) # remove words longer than 15 chars
            apptitle = re.sub(r'[^\s\w_]+','', row[0]) # strip non alphanumeric chars in title

            if (row[2] == 'True'): # if row is a top developer row, separate its content
                top_row = [row[5].strip(), apptitle.strip(), row[19].strip(), row[6].strip(), row[20].strip(), desc.strip()]
                 # Basically write the rows to a list
                dataTop.append(top_row)
            else:           # row isnt top developer, so add to the nontopdev vector
                nontop_row = [row[5].strip(), apptitle.strip(), row[19].strip(), row[6].strip(), row[20].strip(), desc.strip()]
                dataNonTop.append(nontop_row)

        except IndexError as e:
            print e
            pass

    # create new files with the separated lists

    #list of top developers
    with open("data/PlayStoreDataTopDevelopers.csv", "w+") as to_file1:
        writer1 = csv.writer(to_file1, delimiter=",")
        for top_row in dataTop:
            writer1.writerow(top_row)

    #list of non top developers
    with open("data/PlayStoreDataOthers.csv", "w+") as to_file2:
        writer2 = csv.writer(to_file2, delimiter=",")
        for nontop_row in dataNonTop:
            writer2.writerow(nontop_row)
