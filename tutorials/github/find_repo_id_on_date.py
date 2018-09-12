import requests
import numpy as np
from datetime import datetime as dt


def binary_search(max_id, target_time, access_token=None):
    """Returns repo ID closest to target datetime using log(n) search."""
    left_id = 0
    right_id = max_id - 1
    n_steps = 0
    while left_id <= right_id:
        middle_id = int(np.floor(0.5 * (left_id + right_id)))
        proposal_time, proposal_id = get_repo_creation_date(
            middle_id, access_token
        )
        n_steps += 1
        if proposal_time < target_time:
            left_id = middle_id + 1
        elif proposal_time > target_time:
            right_id = middle_id - 1
        else:
            print('Number of search steps was {}.'.format(n_steps))
            return proposal_id, proposal_time
    print('Number of search steps was {}.'.format(n_steps))
    return proposal_id, proposal_time


def get_repo_creation_date(repo_id, access_token=None):
    """Returns the repo creation data given a repo ID using 2 API requests.
    Note that maximum number of requests is 60 without auth token

    """
    # Get the repo info for a given repo id
    q_get_repo_id = 'https://api.github.com/repositories?since=' + \
        str(repo_id - 1)
    if access_token:
        q_get_repo_id += '&' + access_token
    resp = requests.get(q_get_repo_id)

    # Get repo creation date given a repo URL
    q_get_repo = resp.json()[0]['url']
    if access_token:
        q_get_repo += '?' + access_token
    resp = requests.get(q_get_repo)
    repo_id = resp.json()['id']
    repo_created_at = dt.strptime(
        resp.json()['created_at'],
        '%Y-%m-%dT%H:%M:%SZ'
    )

    return repo_created_at, repo_id


def main():
    # Github OAuth access token
    token = 'client_id=7b40beb3e04f09213b27&client_secret=a9a7b427c440338ce5e2d96a181f4cab7e8ebecf'

    # This is the number of created repositories on GitHub as of 2018-09-10
    # including forked repositories.
    max_id = 148194256

    # The dates we are interested in
    preperiod_start = dt(2016, 6, 4)
    intervention_date = dt(2018, 6, 4)
    postperiod_end = dt(2018, 9, 10)
    target_dates = [preperiod_start, intervention_date, postperiod_end]

    for target_date in target_dates:
        repo_id, repo_created_at = binary_search(max_id, target_date, token)
        print('Repo ID {} created at {}.'.format(repo_id, repo_created_at))

if __name__ == "__main__":
    main()
    # Results should be:
    # Repo ID 60384592 created at 2016-06-03 23:59:57.
    # Repo ID 135950139 created at 2018-06-04 00:00:00.
    # Repo ID 148076213 created at 2018-09-10 00:00:00.
