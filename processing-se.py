import os
import random
import re
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup


def processText(input):
    input = input.encode('ascii', 'ignore')
    output = BeautifulSoup(input).text.replace("\n", "").replace("\r", " ").replace("\"", "").replace("\\",
                                                                                                      "\\\\").replace(
        "\t", " ").replace("  ", " ")
    return output


def processTags(input):
    output = re.sub("\s+", input.replace("\n", "").replace("\r", "").replace("<", "").replace(">", ";").replace("\"",
                                                                                                                "").strip(),
                    " ").strip(";")
    output = output.replace("-", " ")
    return output


if __name__ == '__main__':

    data_folder = 'data/stackexchange/raw/'
    train_json_file = 'data/stackexchange/stackexchange_training.json'
    valid_json_file = 'data/stackexchange/stackexchange_validation.json'
    test_json_file = 'data/stackexchange/stackexchange_testing.json'
    post_data_dic = {}

    for filename in os.listdir(data_folder):
        if (filename.startswith("Posts")):
            # print(filename)
            tree = ET.parse(data_folder + filename)
            root = tree.getroot()
            post_dict = {}
            idsuffix = filename.replace("Posts_", "").replace("xml", "")
            for post in root.findall("row"):
                post_dict[post.attrib['Id']] = post.attrib['Body']
            iCount = 0
            for post in root.findall("row"):
                if ('ParentId' not in post.attrib and ('Tags' in post.attrib and 'Body' in post.attrib)):

                    temptext = "{ "
                    temptext = temptext + " \"id\": \"" + idsuffix + post.attrib["Id"] + "\" "
                    if ('Body' in post.attrib):
                        temptext = temptext + " , \"question\": \"" + processText(post.attrib['Body']) + "\" "
                    if ('Title' in post.attrib):
                        temptext = temptext + " , \"title\": \"" + processText(post.attrib['Title']) + "\" "

                    if ('Tags' in post.attrib):
                        temptext = temptext + " , \"tags\": \"" + processTags(post.attrib['Tags']) + "\" "

                    if ('AcceptedAnswerId' in post.attrib and post.attrib['AcceptedAnswerId'] in post_dict):
                        temptext = temptext + " , \"accepted_answer\": \"" + processText(
                            post_dict[post.attrib['AcceptedAnswerId']]) + "\" "

                    temptext += " } "
                    iCount += 1

                    post_data_dic[idsuffix + post.attrib['Id']] = temptext
            print(idsuffix, iCount)

    randomlist = list(post_data_dic.keys())
    random.shuffle(randomlist)

    test = int(int((len(randomlist) * (.05)) / 1000) * 1000)
    # test = 4000
    print(test)
    jsonwriter = open(test_json_file, 'w')
    for id in range(0, test):
        postid = randomlist[id]
        jsonwriter.write(post_data_dic[postid] + "\n")

    jsonwriter = open(valid_json_file, 'w')
    for id in range(test, test * 2):
        postid = randomlist[id]
        jsonwriter.write(post_data_dic[postid] + "\n")

    jsonwriter = open(train_json_file, 'w')
    for id in range(test * 2, len(randomlist)):
        postid = randomlist[id]
        jsonwriter.write(post_data_dic[postid] + "\n")

        #
        # jsonwriter.write(temptext+"\n")
        #             print(temptext)
