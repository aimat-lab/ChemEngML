

def writeToFile(Cf, St, delta, output_dir):
    # Write Cf and St values to a file
    with open(output_dir + "/Cf_St_values.txt", "w") as file:
        file.write("Cf, St, delta\n")
        for cf, st, delta in zip(Cf, St, delta):
            file.write(f"{round(cf,4)}, {round(st,4)}, {round(delta,4)}\n")
            

def writeToFileShap(Cf, St, delta, 
                    Cf_shapSum, St_shapSum, 
                    Cf_backgroundAvg, St_backgroundAvg, 
                    Cf_xtrainAvg, St_xtrainAvg, output_dir):
    # Write Cf and St values to a file
    with open(output_dir + "/Cf_St_valuesWithSHAP.txt", "w") as file:
        file.write("Cf, St, "  +
                   "delta, "  +
                   "Cf_shapSum, St_shapSum, "  +
                   "Cf_backgroundAvg, St_backgroundAvg, "  +
                   "Cf_xtrainAvg, St_xtrainAvg, "  +
                   "Cf_check1, Cf_check2, "  +
                   "St_check1, St_check2\n")
        for Cf, St, delta, Cf_shapSum, St_shapSum in zip(Cf, St, delta, Cf_shapSum, St_shapSum):
            file.write(f"{str(round(Cf,4)):0<6}, {str(round(St,4)):0<6}, "  +
                f"{str(round(delta,4)):0<6}, "  +
                f"{str(round(Cf_shapSum,4)):0<6}, {str(round(St_shapSum,4)):0<6}, "  +
                f"{str(round(Cf_backgroundAvg,4)):0<6}, {str(round(St_backgroundAvg,4)):0<6}, "  +
                f"{str(round(Cf_xtrainAvg,4)):0<6}, {str(round(St_xtrainAvg,4)):0<6}, "  +
                f"{str(round(Cf_shapSum+Cf_backgroundAvg,4)):0<6}, {str(round(Cf_shapSum+Cf_xtrainAvg,4)):0<6}, "  +
                f"{str(round(St_shapSum+St_backgroundAvg,4)):0<6}, {str(round(St_shapSum+St_xtrainAvg,4)):0<6}\n")