import os, sys

TOTALTIME = "total_time_mean"
IOSIZE = "total_io_size"
IOMODE = "io_mode"
DRIVER = "io_driver"
NN = "number_of_nodes"
NP = "number_of_processes"

xname = {NP: "Number of Processes", 
         DRIVER: "IO Driver",
         IOMODE: "IO Mode",
         "bb_blocking_coll": "BB", 
         "bb_private_blocking_coll": "BB-P", 
         "bb_shared_blocking_coll": "BB-S", 
         "ncmpi_blocking_coll": "NC-BC", 
         "ncmpi_nonblocking_coll": "NC-NC", 
         "ncmpi_blocking_indep": "NC-BI", 
         "ncmpi_nonblocking_indep": "NC-NI", 
         "stage_blocking_coll": "DW-BC", 
         "stage_blocking_indep": "DW-BI", 
         "stage_nonblocking_coll": "DW-NC", 
         "stage_nonblocking_indep": "DW-NI", 
         "blocking_coll": "BC", 
         "blocking_indep": "BI", 
         "nonblocking_coll": "NC", 
         "nonblocking_indep": "NI", 
         256: "256",
         512: "512",
         1024: "1024",
         2048: "2048",
         4096: "4096",
         8192: "8192",
        }


def gather(dir:str):
    recs = []
    for filename in os.listdir(dir):
        if filename.endswith(".txt"):
            path = os.path.join(dir, filename)
            rec = {}
            currec = None
            drop = False
            with open(path, 'r') as fin:
                stageouttime = 0
                for line in fin:
                    if (line[:40] == '-----+-----++------------+++++++++--+---'):
                        if (rec[DRIVER] == "stage"):
                            if rec["stage_time"] == 0:
                                rec["stage_time"] = stageouttime
                                rec["total_time_mean"] = rec["flash_time_mean"] + stageouttime
                                rec["total_time_min"] = rec["flash_time_mean"] + stageouttime
                                rec["total_time_max"] = rec["flash_time_mean"] + stageouttime
                            else:
                                stageouttime = rec["stage_time"]
                        recs.append(rec)
                        rec = {}
                    elif (line[:3] == '#%$'):
                        tokens = line.split(sep = ':')
                        val = tokens[2].strip()
                        try:
                            int(val)
                            val = int(val)
                        except ValueError:
                            try:
                                float(val)
                                val = float(val)
                            except ValueError:
                                val = val
                        rec[tokens[1].strip()] = val
                    elif (line[:5] == 'Error' or line[:7] == '#Error!'):
                        print(line)
                        drop = True
    return recs

def plot_driver_np(fout, recs:list, filter:dict):
    cnt = {}
    time = {}
    size = {}
    xs = set()
    ys = set()
    for rec in recs:
        drop = False
        for k in filter:
            if (k not in rec or rec[k] != filter[k]):
                drop = True
                break
        if not drop:
            x = rec[NP]
            y = rec[DRIVER] + "_" + rec[IOMODE]
            k = str(x) + "/" + str(y)
            xs.add(x)
            ys.add(y)
            if k not in cnt:
                cnt[k] = 1
                time[k] = rec[TOTALTIME]
                size[k] = rec[IOSIZE]
            else:
                cnt[k] += 1
                if (time[k] > rec[TOTALTIME]):
                    time[k] = rec[TOTALTIME]
                    size[k] = rec[IOSIZE]
    xs = sorted(list(xs))
    ys = sorted(list(ys))
    
    fout.write("End to End IO Time,	Time (sec), ")
    for k in filter:
        fout.write(str(k) + ": " + str(filter[k]) + ",")
    fout.write("\n")
    fout.write(xname[NP] + ", ")
    for y in ys:
        fout.write(str(xname[y]) + ", ")
    fout.write("\n")
    for x in xs:
        fout.write(str(xname[x]) + ", ")
        for y in ys:
            k = str(x) + "/" + str(y)
            if (k in time):
                fout.write(str(time[k]))
            fout.write(", ")
        fout.write("\n")

    fout.write("GiB, ")
    for k in filter:
        fout.write(str(k) + ": " + str(filter[k]) + ",")
    fout.write("\n")
    fout.write(xname[NP] + ", ")
    for y in ys:
        fout.write(str(xname[y]) + ", ")
    fout.write("\n")
    for x in xs:
        fout.write(str(xname[x]) + ", ")
        for y in ys:
            k = str(x) + "/" + str(y)
            if (k in size):
                fout.write(str(size[k]))
            fout.write(", ")
        fout.write("\n")

    fout.write("End to End IO Bandwidth, Bandwidth (GiB/s), ")
    for k in filter:
        fout.write(str(k) + ": " + str(filter[k]) + ",")
    fout.write("\n")
    fout.write(xname[NP] + ", ")
    for y in ys:
        fout.write(str(xname[y]) + ", ")
    fout.write("\n")
    for x in xs:
        fout.write(str(xname[x]) + ", ")
        for y in ys:
            k = str(x) + "/" + str(y)
            if (k in time and k in size):
                fout.write(str(size[k] / time[k]))
            fout.write(", ")
        fout.write("\n")

    fout.write("Runs, ")
    for k in filter:
        fout.write(str(k) + ": " + str(filter[k]) + ",")
    fout.write("\n")
    fout.write(xname[NP] + ", ")
    for y in ys:
        fout.write(str(xname[y]) + ", ")
    fout.write("\n")
    for x in xs:
        fout.write(str(xname[x]) + ", ")
        for y in ys:
            k = str(x) + "/" + str(y)
            if (k in cnt):
                fout.write(str(cnt[k]))
            fout.write(", ")
        fout.write("\n")

def plot_stage_np_mode(fout, recs:list, filter:dict):
    table = {}
    cnt = {}
    xs = set()
    x2s = set()
    ys = set()
    for rec in recs:
        drop = False
        for k in filter:
            if (k not in rec or rec[k] != filter[k]):
                drop = True
                break
        if not drop:
            x = rec[NP]
            x2 = rec[IOMODE]
            xs.add(x)
            x2s.add(x2)
            k = str(x) + "_" + str(x2)
            if k not in table:
                table[k] = rec
                cnt[k] = 1
            else:
                cnt[k] += 1
                if (table[k][TOTALTIME] > rec[TOTALTIME]):
                    table[k] = rec

    xs = sorted(list(xs))
    x2s = sorted(list(x2s))
    ys = ["total_time_mean", "flash_time_mean", "stage_time"]
    xys = {"total_time_mean": "Total Time", "flash_time_mean": "Time Writting to BB", "stage_time": "Time Staging Out"}

    fout.write("DW Stagin Time Breakdown, Time (Sec), ")
    for k in filter:
        fout.write(str(k) + ": " + str(filter[k]) + ",")
    fout.write("\n")
    fout.write(xname[NP] + ", " + IOMODE + ", ")
    for y in ys:
        fout.write(str(xys[y]) + ", ")
    fout.write("\n")
    for x in xs:
        fout.write(str(xname[x]))
        for x2 in x2s:
            fout.write(", " + xname[x2] + ", ")
            k = str(x) + "_" + str(x2)
            if (k in table):
                for y in ys:
                    if (y in table[k]):
                        fout.write(str(table[k][y]))
                    fout.write(", ")
            fout.write("\n")

    fout.write("Runs, ")
    for k in filter:
        fout.write(str(k) + ": " + str(filter[k]) + ",")
    fout.write("\n")
    fout.write(xname[NP] +  ", " + IOMODE + ", ")
    for y in ys:
        fout.write(str(xys[y]) + ", ")
    fout.write("\n")
    for x in xs:
        fout.write(str(xname[x]))
        for x2 in x2s:
            fout.write(", " + xname[x2] + ", ")
            k = str(x) + "_" + str(x2)
            if (k in table):
                for y in ys:
                    if (y in table[k]):
                        fout.write(str(cnt[k]))
                    fout.write(", ")
            fout.write("\n")

def plot_bb_np_cmp(fout, recs:list, filter:dict):
    table = {}
    cnt = {}
    xs = set()
    x2s = set()
    ys = set()
    for rec in recs:
        drop = False
        for k in filter:
            drop = True
            if (k in rec):
                for f in list(filter[k]):
                    if rec[k] == f:
                        drop = False
            if drop:
                break
        if not drop:
            x = rec[NP]
            x2 = rec[DRIVER]
            xs.add(x)
            x2s.add(x2)
            k = str(x) + "/" + str(x2)
            if k not in table:
                table[k] = rec
                cnt[k] = 1
            else:
                cnt[k] += 1
                if (table[k][TOTALTIME] > rec[TOTALTIME]):
                    table[k] = rec

    xs = sorted(list(xs))
    x2s = sorted(list(x2s))
    
    ys = ["bb_total_time_mean", "bb_create_time_mean", "bb_enddef_time_mean", "bb_put_time_mean", "bb_flush_time_mean", "bb_close_time_mean", 
          "bb_put_data_wr_time_mean", "bb_put_meta_wr_time_mean", "bb_put_num_wr_time_mean", 
          "bb_flush_replay_time_mean", "bb_flush_data_rd_time_mean", "bb_flush_put_time_mean", "bb_flush_wait_time_mean"]
    xys = {"bb_total_time_mean": "Time in DW Driver", "bb_create_time_mean": "Time Creating Log", "bb_enddef_time_mean": "Time Enddef", "bb_put_time_mean": "Time Writing Log", "bb_flush_time_mean": "Time Flushing Log", "bb_close_time_mean": "Time Closing Log", 
           "bb_put_data_wr_time_mean": "Time Writing Data", "bb_put_meta_wr_time_mean": "Time Writing Metadata", "bb_put_num_wr_time_mean": "Time Updateing Num", 
           "bb_flush_replay_time_mean": "Time Replaying Log", "bb_flush_data_rd_time_mean": "Time Reading Data", "bb_flush_put_time_mean": "Time Calling Put", "bb_flush_wait_time_mean": "Time Waiting"}
    xx2s = {"bb": "BB", 
            "bb_shared": "BB-S", 
            'bb_private': "BB-P"}

    fout.write("DW Driver Time Breakdown, Time (Sec), ")
    for k in filter:
        fout.write(str(k) + ": " + str(filter[k]) + ",")
    fout.write("\n")
    fout.write(xname[NP] +  ", " + xname[DRIVER] + ", ")
    for y in ys:
        fout.write(str(xys[y]) + ", ")
    fout.write("\n")
    for x in xs:
        fout.write(str(xname[x]))
        for x2 in x2s:
            fout.write(", " + str(xx2s[x2]) + ", ")
            k = str(x) + "/" + str(x2)
            if k in table:
                for y in ys:
                    if (y in table[k]):
                        fout.write(str(table[k][y]))
                    fout.write(", ")
            fout.write("\n")

    fout.write("Runs, ")
    for k in filter:
        fout.write(str(k) + ": " + str(filter[k]) + ",")
    fout.write("\n")
    fout.write(xname[NP] +  ", " + xname[DRIVER] + ", ")
    for y in ys:
        fout.write(str(xys[y]) + ", ")
    fout.write("\n")
    for x in xs:
        fout.write(str(xname[x]))
        for x2 in x2s:
            fout.write(", " + str(xx2s[x2]) + ", ")
            k = str(x) + "/" + str(x2)
            if k in table:
                for y in ys:
                    if (y in table[k]):
                        fout.write(str(cnt[k]))
                    fout.write(", ")
            fout.write("\n")

def main(argv:list):
    dir = 'C:/Users/x3276/OneDrive/Research/Log io/Result/flash/theta'

    if len(argv) > 1:
        dir = argv[1]
    
    recs = gather(dir)
    
    with open('flash.csv', 'w') as fout:
        filter = {}
        plot_driver_np(fout, recs, filter)
        filter = {DRIVER: ['bb', 'bb_shared', 'bb_private']}
        plot_bb_np_cmp(fout, recs, filter)
        filter = {DRIVER: 'stage'}
        plot_stage_np_mode(fout, recs, filter)

        
if __name__=='__main__':
    main(sys.argv)