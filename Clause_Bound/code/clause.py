import nltk
import re
from nltk.tree import *
import sys
sys.stdout = open("hindi.txt", "w")

# -----------------------------Different parse tree structures--------------------------------


# ---------------Hindi Sentences -----------------


parse_tree="(TOP(S(SBAR(SBAR(WHADVP (SCONJ When))(S (NP (PRON I)) (VP (VERB came) (ADVP (ADV here)))))(PUNCT ,)(NP (PRON I))(VP (VERB saw) (NP (PRON him))))(PUNCT ,)(CCONJ and)(S (NP (PRON he)) (VP (VERB greeted) (NP (PRON me))))))"


# parse_tree="(TOP(NP(NP (NP (PROPN श्याम) (NOUN घर)) (NP (VERB गया)))(NP (NP (PROPN राम)) (PART नहीं) (VP (VERB गया)))))"


# parse_tree="(TOP(S(NP (NP (PRON यह)) (NP (DET वही) (NOUN लड़का)))(VP(AUX है)(SBAR(WHNP (PRON जिसने))(NP(NP (NOUN कल) (NOUN चोरी))(VP (VERB की) (AUX थी) (VP (PUNCT |))))))))"

# parse_tree="""(TOP(NP(NP (NP (PROPN श्याम) (NOUN घर)) (NP (VERB गया)))(NP (NP (PROPN राम)) (PART नहीं) (VP (VERB गया)))))"""

# parse_tree="(TOP(S(NP(NP(NP (NP (PRON वह)) (NOUN लड़का))(WHNP (PRON जो))(VERB खेल))(AUX रहा)(SBAR (AUX था)))(NP (NOUN घर))(VP (VERB गया))))"


# parse_tree="(TOP(NP (NP (PRON यह)) (DET वही) (NOUN व्यक्ति))(AUX है)(WHNP (WHNP (PRON जिसकी)) (NP (NOUN कल)))(VP (NP (VP (VERB पिटाई)) (VERB की)) (FRAG (NP (AUX गई)))(AUX थी)))"

# parse_tree="(TOP(VP (NP (NP (PROPN राम)) (NP (NOUN खाना))) (VP (VERB खाकर)))(CCONJ और)(NP(NP (NP (NOUN पानी)) (VP (VERB पीकर)))(NP (NP (NOUN घर)) (VP (VERB गया)))))"

# parse_tree="(TOP(NP (PROPN राम))(SBAR (WHNP (PRON जिसने)) (NP (NP (NOUN खाना)) (VP (VERB खाया))))(CCONJ और)(NP(NP (NOUN खेल) (NOUN खेला))(NP (NP (NOUN घर)) (VP (VERB गया)))))"

# parse_tree="""(TOP(NP(NP(NP(NP (NP (PRON वह)) (NOUN लड़का))(WHNP (PRON जो))(VERB खेल))(AUX रहा)(SBAR (AUX था)))(NP (NOUN घर))(VP (VERB गया))))"""

# parse_tree="(TOP(NP(IP(NP (PROPN राम))(VP (ADJP (QP (ADP ने) (NP (NOUN काम)))) (VP (VERB किया))))(CCONJ और)(NP (NOUN खाना) (VERB खाया))(CCONJ लेकिन)(NP (NP (PROPN सीता)) (VERB खेली))))"

# parse_tree="(TOP (S (PRON मैंने))(VP(VERB कहा)(SBAR(SCONJ कि)(IP(NP (PRON मैंने))(NP(DET यह)(NP(NOUN कलम)(VP(VERB खरीदी)(AUX है)(SBAR(WHNP (PRON जो))(IP(NP (NP (NOUN बाजार)) (ADP में))(NP (ADV सबसे) (NP (NP (ADJ सस्ती)) (VP (AUX है)))))))))))))"

# ---------------English Sentences ---------
# parse_tree="(TOP(S(SBAR(SCONJ Though)(S(NP (PRON he))(VP (AUX was) (ADJP (ADV very) (ADJ rich)))))(PUNCT ,)(NP (PRON he))(VP (AUX was)(ADVP (ADV still))(ADJP (ADV very) (ADJ unhappy)))))"
#
# parse_tree="(TOP(S(S((NP (PRON I))VP(VERB looked)(PP (ADP for) (NP (PROPN Ram) (CCONJ and) (PROPN Shyam)))(PP (ADP at) (NP (DET the) (NOUN railway) (NOUN station)))))(PUNCT ,)(CCONJ but)(S(NP (PRON they))(VP(VP(VERB arrived)(PP (ADP at) (NP (DET the) (NOUN station)))(PP (ADP before) (NP (NOUN noon))))(CCONJ and)(VP(VERB left)(PP (ADP on) (NP (DET the) (NOUN train)))(SBAR (ADP before) (S (NP (PRON I)) (VP (VERB arrived)))))))(PUNCT .)))"

# parse_tree="""(TOP(S(NP(NP (DET The) (NOUN police) (NOUN officer))(PP (ADP of) (NP (PROPN Hamirpur) (PROPN District))))(VP(VERB announced)(SBAR(SCONJ that)(S(NP (PROPN alchol))(VP(AUX had)(VP(VERB declined)(NP (NUM 80) (NOUN percent))(PP(ADP in)(NP(NP (PROPN College))(PUNCT ,)(SBAR(WHADVP (SCONJ whereas))(S(NP (PRON there))(VP(AUX had)(VP(VERB been)(NP(NP (DET a) (ADJ big) (NOUN jump))(PP(ADP in)(NP(NP (DET the) (NOUN number))(PP (ADP of) (NP (NOUN cigarette)))))))))))))))))))"""

# parse_tree="(TOP(S(NP (DET The) (NOUN horse))(VP (VERB went) (PP (ADP to) (NP (DET the) (NOUN ground))))))"

# parse_tree="(TOP(S(NP (PROPN Rohit))(VP(VP (VERB likes) (NP (NOUN music)))(CCONJ but)(VP (AUX does) (PART not) (VP (VERB like) (NP (NOUN drums)))))))"

# parse_tree="(TOP(S(NP (PROPN Joe))(VP(VERB realized)(SBAR(SCONJ that)(VP (NP (DET the) (NOUN train)) (AUX was) (ADJP (ADJ late))))(SBAR(SCONJ while)(S(NP (PRON he))(VP(VERB waited)(PP (ADP at) (NP (DET the) (NOUN train) (NOUN station)))))))))"

# parse_tree="(TOP(S(NP (DET Every) (NOUN night))(NP (DET the) (NOUN office))(VP(AUX is)(VP(VP (VERB dusted))(CCONJ and)(VP(VERB mopped)(PP (ADP by) (NP (DET the) (NOUN janitors))))))))"
# ------------------------------------------------------------------------------------------------


# Clause = SubjectPhrase + VerbPhrase
# Find the position of first node "VP" while traversing from top to bottom and also find the position of subordinating conjunctions.
# Delete the nodes at these positions to get the subject by first deleteting the VP nodes then subordinating conjunction nodes
# ie The part without VP and subordinating conjunction nodes ie Subject Phrase

def vp_conj_position(t):
    # print(t)
    vp_pos = []
    sub_conj_pos = []
    num_children = len(t)

    # print(num_children)
    # for i in range(0, num_children):
        # print(t[i])
        # print(t[i].label())

    children = [t[i].label() for i in range(0, num_children)]

    # print("Children : ",children)

    flag = re.search(r"(SBARQ|SQ|SBAR|SINV|S)", ' '.join(children))

    if "VP" in children and not flag:
        for i in range(0, num_children):
            if t[i].label() == "VP":
                vp_pos.append(t[i].treeposition())
    elif not "VP" in children and not flag:
        for i in range(0, num_children):
            if t[i].height() > 2:
                temp1, temp2 = vp_conj_position(t[i])
                vp_pos.extend(temp1)
                sub_conj_pos.extend(temp2)
    else:
        for i in range(0, num_children):
            if t[i].label() in ["S", "SBAR", "SBARQ", "SINV", "SQ"]:
                temp1, temp2 = vp_conj_position(t[i])
                vp_pos.extend(temp1)
                sub_conj_pos.extend(temp2)
            else:
                sub_conj_pos.append(t[i].treeposition())

    return (vp_pos, sub_conj_pos)



def get_verb_phrases(t):

    # print("Input to get get VP : ",t)

    verb_phrases = []
    num_children = len(t)

    # print("Length of input: ",num_children)
    # for i in range(0, num_children):
    #     print(t[i])
    #     print(t[i].label())

    num_VP = sum(1 if (t[i].label() == "VP" ) else 0 for i in range(0, num_children))

    # print("Number of VP :",num_VP)

    if t.label() != "VP":
        for i in range(0, num_children):
            if t[i].height() > 2:
                verb_phrases.extend(get_verb_phrases(t[i]))

    elif t.label() == "VP" and num_VP > 1:
        for i in range(0, num_children):
            if t[i].label() == "VP":
                if t[i].height() > 2:
                    verb_phrases.extend(get_verb_phrases(t[i]))

    else:
        verb_phrases.append(' '.join(t.leaves()))

    return verb_phrases



# Clause level list from Penn Treebank II = ["S", "SBAR", "SBARQ", "SINV", "SQ"]

def print_clauses(parse_str):
    parented_tree = ParentedTree.fromstring(str(parse_str))
    clause_level_list = ["S", "SBAR", "SBARQ", "SINV", "SQ"]
    clause_list = []
    acceptable_sub_trees = []


    sentence = ' '.join(parented_tree.leaves())


    print("Sentence : ",sentence)
    print("\n POS Tags: ",parented_tree.pos())
    print("\n Parse Tree : ", parse_str)
    print("\n -----------------Parented Tree----------------- \n",parented_tree)
    print("\n ------------------------------------------------ \n")
    print("\n ----------------Constituency Parse Tree------------------- \n")
    parented_tree.pretty_print()
    print("\n ----------------------------------------------------------- \n")

    # To draw in new window
    # parented_tree.draw()


    # Breaking tree into subtrees of clauses using
    # clause levels "S","SBAR","SBARQ","SINV","SQ"

    # label(s) is NP => label(s.parent) is ROOT => Add leaves of s to clause list
    # label(s) in CLL => label(s.parent) in CLL => Yes => go to next subtree.
    # label(s) in CLL => label(s.parent) in CLL => No =>Check len(s)=1 and label(s)=S and label(s[0])=VP and label(s.parent) not in CLL => If true go to next subtree. If False => add s to ASL and selete s from allsubtrees.

    all_subtrees=list(parented_tree.subtrees())
    print("Number of Subtrees possible : ",len(all_subtrees))

    for s in reversed(list(all_subtrees)):
        # s.pretty_print()
        # print("label : ",s.label() , "Parent label :",s.parent().label(),"Parent's parent :",s.parent().parent().label())

        # if s.label() =="NP":

        #     if (s.parent().label()=="TOP" or s.parent().label=="ROOT"):
        #         # print(sub_tree)
        #         subject_phrase = ' '.join(s.leaves())
        #         clause_list.append(subject_phrase)

        #     if (s.parent().label() =="NP" or s.parent().label()=="TOP" or s.parent().label=="ROOT"):
        #         # print(sub_tree)
        #         if(s.parent().label() =="NP" and s.parent().parent().label()=="TOP"):
        #             subject_phrase = ' '.join(s.leaves())
        #             clause_list.append(subject_phrase)
        #             continue

        if s.label() in clause_level_list:
            if s.parent().label() in clause_level_list:
                continue

            if (len(s) == 1 and s.label() == "S" and s[0].label() == "VP" and not s.parent().label() in clause_level_list):

                continue

            acceptable_sub_trees.append(s)
            # print(parented_tree[sub_tree.treeposition()])
            del parented_tree[s.treeposition()]


    print("Number of Accepted Subtrees: ",len(acceptable_sub_trees))



    count=0;
    for t in acceptable_sub_trees:

        count+=1
        print("\n--------Accepted Subtree ",count,"----------\n")
        print("\n Leaf Nodes : ",t.leaves())
        t.pretty_print()

        # get verb phrases from the new modified tree
        verb_phrases = get_verb_phrases(t)
        print("\nVerb Phrases in Subtree ",count," : ",verb_phrases)

        vp_pos, sub_conj_pos = vp_conj_position(t)
        print("Positions of VerbPhrase and Subordinationg Conjunctions : ",(vp_pos, sub_conj_pos))

        # Get tree without verb phrases and subordinating conjunctions nodes (mainly subject)

        for i in reversed(vp_pos):
            print(" Deleting Verb Phrase :  " , t[i].leaves())
            # t[i].pretty_print()
            del t[i]

        for i in reversed(sub_conj_pos):
            print(" Deleting Sub. Ordinating Conj Phrase : " , t[i].leaves())
            # t[i].pretty_print()
            del t[i]

        print("\n Tree Left After Deletion: ")
        t.pretty_print()

        subject_phrase = ' '.join(t.leaves())
        print("\n Subject Phrase : ",subject_phrase)


        # Updating the clause_list by joining subject and verb phrases
        for i in verb_phrases:
            clause_list.append(subject_phrase + " " + i)

        print("\n--------Accepted Subtree ",count," End----------\n")

    for i in clause_list:
        print(len((i).split()))
        if len(i.split())==1:
            clause_list.remove(i)

    print("Clauses : ",clause_list)
    return clause_list


print_clauses(parse_tree)
