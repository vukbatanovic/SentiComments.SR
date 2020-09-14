import krippendorff
import numpy as np


def print_statistics(name, agreements, alphas):
    print(name + ' agreement / Krippendorff\'s Alpha: ')
    print('IG\tEG\tCG')
    for i in range(0, 3):
        print(format(agreements[i], '.3f') + '/' + format(alphas[i], '.3f') + '\t', end='')
    print()


def print_inter_group_statistics(name, alphas):
    print(name + ': ')
    print('IG & EG\tIG & CG\tEG & CG')
    for i in range(0,3):
        print(format(alphas[i], '.3f') + '\t', end='')
    print()


def map_to_int(scores):
    mapped_scores = []
    for s in scores:
        mapped_scores.append(string_score_map[s])
    return mapped_scores


gradesPolarity = []
gradesSubjectivity = []
gradesFourWay = []
gradesSixWay = []
gradesSarcasm = []
agreementsPolarity = []
alphasPolarity = []
agreementsSubjectivity = []
alphasSubjectivity = []
agreementsFourWay = []
alphasFourWay = []
agreementsSixWay = []
alphasSixWay = []
agreementsSarcasm = []
alphasSarcasm = []
alphasPolarityInterGroup = []
alphasSubjectivityInterGroup = []
alphasFourWayInterGroup = []
alphasSixWayInterGroup = []
alphasSarcasmInterGroup = []
for i in range(0, 6):
    gradesPolarity.append([])
    gradesSubjectivity.append([])
    gradesFourWay.append([])
    gradesSixWay.append([])
    gradesSarcasm.append([])

string_score_map = {'+': 1001, '-': -1000, '+1': 11, '-1': -10, 'NS': 0, '+NS': 10000, '-NS':-105, 'M': 55, '+M': 560, '-M': -50}

with open('SentiComments.SR.verif.books.txt', 'r', encoding='utf-8') as annofile:
    for line in annofile:
        parts = line.strip().split('\t')
        grades = parts[:-1]
        for i in range(0, 6):
            grade = grades[i].strip()
            if grade[0] != '+' and grade[0] != '-':
                grade = grade[1:]
            if grade[-1] == 's':
                gradesSixWay[i].append(grade[0:-1])
                gradesSarcasm[i].append(1)
            else:
                gradesSixWay[i].append(grade)
                gradesSarcasm[i].append(0)
            gradesPolarity[i].append(grade[0])
            if grade[1:3] != 'NS':
                gradesSubjectivity[i].append(1)
                if grade[1] == 'M':
                    gradesFourWay[i].append('M')
                elif grade[:2] == '+1':
                    gradesFourWay[i].append('+1')
                else:
                    gradesFourWay[i].append('-1')
            else:
                gradesSubjectivity[i].append(0)
                gradesFourWay[i].append('NS')

for i in range(0, 3):
    agreementsPolarity.append(
        float(sum(np.array(gradesPolarity[2*i]) == np.array(gradesPolarity[2*i+1]))) / len(gradesPolarity[2*i]))
    alphasPolarity.append(krippendorff.alpha(reliability_data=[map_to_int(gradesPolarity[2 * i]), map_to_int(gradesPolarity[2 * i + 1])], level_of_measurement='nominal'))
    agreementsSubjectivity.append(
        float(sum(np.array(gradesSubjectivity[2*i]) == np.array(gradesSubjectivity[2*i+1]))) / len(gradesSubjectivity[2*i]))
    alphasSubjectivity.append(krippendorff.alpha(reliability_data=[gradesSubjectivity[2 * i], gradesSubjectivity[2 * i + 1]], level_of_measurement='nominal'))
    agreementsFourWay.append(
        float(sum(np.array(gradesFourWay[2*i]) == np.array(gradesFourWay[2*i+1]))) / len(gradesFourWay[2*i]))
    alphasFourWay.append(krippendorff.alpha(reliability_data=[map_to_int(gradesFourWay[2 * i]), map_to_int(gradesFourWay[2 * i + 1])], level_of_measurement='nominal'))
    agreementsSixWay.append(
        float(sum(np.array(gradesSixWay[2*i]) == np.array(gradesSixWay[2*i+1]))) / len(gradesSixWay[2*i]))
    alphasSixWay.append(krippendorff.alpha(reliability_data=[map_to_int(gradesSixWay[2 * i]), map_to_int(gradesSixWay[2 * i + 1])], level_of_measurement='nominal'))
    agreementsSarcasm.append(
        float(sum(np.array(gradesSarcasm[2*i]) == np.array(gradesSarcasm[2*i+1]))) / len(gradesSarcasm[2*i]))
    alphasSarcasm.append(krippendorff.alpha(reliability_data=[gradesSarcasm[2 * i], gradesSarcasm[2 * i + 1]], level_of_measurement='nominal'))

alphasPolarityInterGroup.append(krippendorff.alpha(reliability_data=[map_to_int(gradesPolarity[0]), map_to_int(gradesPolarity[1]), map_to_int(gradesPolarity[2]), map_to_int(gradesPolarity[3])], level_of_measurement='nominal'))
alphasPolarityInterGroup.append(krippendorff.alpha(reliability_data=[map_to_int(gradesPolarity[0]), map_to_int(gradesPolarity[1]), map_to_int(gradesPolarity[4]), map_to_int(gradesPolarity[5])], level_of_measurement='nominal'))
alphasPolarityInterGroup.append(krippendorff.alpha(reliability_data=[map_to_int(gradesPolarity[2]), map_to_int(gradesPolarity[3]), map_to_int(gradesPolarity[4]), map_to_int(gradesPolarity[5])], level_of_measurement='nominal'))
alphasSubjectivityInterGroup.append(krippendorff.alpha(reliability_data=[gradesSubjectivity[0], gradesSubjectivity[1], gradesSubjectivity[2], gradesSubjectivity[3]], level_of_measurement='nominal'))
alphasSubjectivityInterGroup.append(krippendorff.alpha(reliability_data=[gradesSubjectivity[0], gradesSubjectivity[1], gradesSubjectivity[4], gradesSubjectivity[5]], level_of_measurement='nominal'))
alphasSubjectivityInterGroup.append(krippendorff.alpha(reliability_data=[gradesSubjectivity[2], gradesSubjectivity[3], gradesSubjectivity[4], gradesSubjectivity[5]], level_of_measurement='nominal'))
alphasFourWayInterGroup.append(krippendorff.alpha(reliability_data=[map_to_int(gradesFourWay[0]), map_to_int(gradesFourWay[1]), map_to_int(gradesFourWay[2]), map_to_int(gradesFourWay[3])], level_of_measurement='nominal'))
alphasFourWayInterGroup.append(krippendorff.alpha(reliability_data=[map_to_int(gradesFourWay[0]), map_to_int(gradesFourWay[1]), map_to_int(gradesFourWay[4]), map_to_int(gradesFourWay[5])], level_of_measurement='nominal'))
alphasFourWayInterGroup.append(krippendorff.alpha(reliability_data=[map_to_int(gradesFourWay[2]), map_to_int(gradesFourWay[3]), map_to_int(gradesFourWay[4]), map_to_int(gradesFourWay[5])], level_of_measurement='nominal'))
alphasSixWayInterGroup.append(krippendorff.alpha(reliability_data=[map_to_int(gradesSixWay[0]), map_to_int(gradesSixWay[1]), map_to_int(gradesSixWay[2]), map_to_int(gradesSixWay[3])], level_of_measurement='nominal'))
alphasSixWayInterGroup.append(krippendorff.alpha(reliability_data=[map_to_int(gradesSixWay[0]), map_to_int(gradesSixWay[1]), map_to_int(gradesSixWay[4]), map_to_int(gradesSixWay[5])], level_of_measurement='nominal'))
alphasSixWayInterGroup.append(krippendorff.alpha(reliability_data=[map_to_int(gradesSixWay[2]), map_to_int(gradesSixWay[3]), map_to_int(gradesSixWay[4]), map_to_int(gradesSixWay[5])], level_of_measurement='nominal'))
alphasSarcasmInterGroup.append(krippendorff.alpha(reliability_data=[gradesSarcasm[0], gradesSarcasm[1], gradesSarcasm[2], gradesSarcasm[3]], level_of_measurement='nominal'))
alphasSarcasmInterGroup.append(krippendorff.alpha(reliability_data=[gradesSarcasm[0], gradesSarcasm[1], gradesSarcasm[4], gradesSarcasm[5]], level_of_measurement='nominal'))
alphasSarcasmInterGroup.append(krippendorff.alpha(reliability_data=[gradesSarcasm[2], gradesSarcasm[3], gradesSarcasm[4], gradesSarcasm[5]], level_of_measurement='nominal'))


print('Intra-group statistics')
print_statistics('Polarity', agreementsPolarity, alphasPolarity)
print_statistics('Subjectivity', agreementsSubjectivity, alphasSubjectivity)
print_statistics('Four-way', agreementsFourWay, alphasFourWay)
print_statistics('Six-way', agreementsSixWay, alphasSixWay)
print_statistics('Sarcasm', agreementsSarcasm, alphasSarcasm)
print()

print('Inter-group statistics - Krippendorff\'s Alpha')
print_inter_group_statistics('Polarity', alphasPolarityInterGroup)
print_inter_group_statistics('Subjectivity', alphasSubjectivityInterGroup)
print_inter_group_statistics('Four-way', alphasFourWayInterGroup)
print_inter_group_statistics('Six-way', alphasSixWayInterGroup)
print_inter_group_statistics('Sarcasm', alphasSarcasmInterGroup)
