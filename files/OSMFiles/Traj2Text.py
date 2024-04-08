import numpy as np

living_zones = np.load("living_zones.npy")
travel_zones = np.load("travel_zone.npy")

def gps_to_zone(gps, zones):
    return zones[int((gps[0] - 116.2075) * 100) * 30 + int((gps[1] - 39.7523) * 100)]


def calculate_times(traj):
    living_times = {}
    travel_times = {}

    for i in range(1, len(traj)):
        st_time = traj[0]
        end_time = traj[i]
        now_gps = traj[2]

        # if st_time and end_time are between 0:00-8:00 or st_time and end_time are between 22:00-24:00
        zone = gps_to_zone(now_gps, living_zones)
        if zone not in living_times:
            living_times[zone] = 0
        living_times[zone] += end_time - st_time

        # if st_time and end_time are between 8:00-22:00
        zone = gps_to_zone(now_gps, travel_zones)
        if zone not in travel_times:
            travel_times[zone] = 0
        travel_times[zone] += end_time - st_time

    # calculate the most of time in living zone
    max_time = 0
    max_zone = 0
    for zone in living_times:
        if living_times[zone] > max_time:
            max_time = living_times[zone]
            max_zone = zone

    if max_time < 60 * 6:
        return -1

    go_to_zones = []
    for zone in travel_times:
        if travel_times[zone] > 60 * 2:
            go_to_zones.append(zone)

    if len(go_to_zones) == 0:
        return -1

    now_str = "I live in " + str(max_zone) + "."
    go_str = "I want go to "
    for zone in go_to_zones:
        go_str += str(zone) + ", "
    go_str = go_str[:-2] + "."

    return now_str + " " + go_str
