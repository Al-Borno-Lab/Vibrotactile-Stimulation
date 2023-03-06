# Vibrotactile-Stimulation

## Announcement: 

## Update Log:
- 2023-03-05 - README draft 1 uploaded.

## Notice:
- .ipynb notebooks are not tracked / synced in git.
- Following directories can be used to ignore files (files in them will not be tracked):
    - `ignored_dir`

## How to use this codebase?
- The codebase is built as a Python package with subpackages for different use.
- Model: The LIF Network is defined under `src.model.lifNetwork`
- Plotting functions: The plotting functions are defined under `src.plotting.plotStructure`
- Reinforcement learning: **Work in Progress**
- Import each of the module as one would with any other Python packages.
    - Ex: `from src.model import lifNetwork as lif`
    - Ex: `from src.plotting import plotStructure as lifplt`

## How to collaborate?
The main branch of the repository maintains the most up-to-date working code, with each branch for development purpose. The branching process allows for us to keep track of changes and maintain a versioned history if we need to go back in the change history.

Due to Github's [removal of password authentication process on August 13, 2021](https://github.blog/2020-12-15-token-authentication-requirements-for-git-operations/), the most convenient way to utilize Github on your local computer is with the [GitHub Desktop](https://desktop.github.com/) client.
It is a open-source tool that also manages your Github credential and makes using git much enjoyable.

### Installing Github Desktop:
- [Github Desktop Installation Instructions](https://docs.github.com/en/desktop/installing-and-configuring-github-desktop/installing-and-authenticating-to-github-desktop/installing-github-desktop)

### Tutorials on Github Desktop:
Below are recommended tutorials to familiarize oneself with:
1. [Cloning and Forking with Github Desktop](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/adding-and-cloning-repositories/cloning-and-forking-repositories-from-github-desktop)
2. [Branches](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/making-changes-in-a-branch/managing-branches)
3. [Commits](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/managing-commits)
4. [Syncing Branches](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/keeping-your-local-repository-in-sync-with-github/syncing-your-branch)
5. [Creating an Issue or Pull Request](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/working-with-your-remote-repository-on-github-or-github-enterprise/creating-an-issue-or-pull-request)
6. [Pushing](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/making-changes-in-a-branch/pushing-changes-to-github)

### Collaboration Process:
`pull > create/select branch > add & commit > push`
- To collaborate, please follow the typical git workflow:
    - "Pull" - Pull the repository from Github to make sure your local copy is up to date. 
    - "Create/Select Branch" - The branching process forks a revision tracking from the main-branch. Name it with the something descriptive of the name of the issue that you're working on.
    - "Add & Commit" - Commit the changes to git at each major milestone of your development cycle on the branch, and give it a meaningful commit-message. Consider commits as snapshots of your progress and will be the timepoints that we can revert back to if something goes wrong.
    - "Push" - Push your local progress to Github. One can push to Github after ANY commit, and frequent "PUSH" makes sure the online (i.e., Github) copy to be reflective of your progress so that when others PULL from Github to collaborate, they have the most recent copy of your code. 
        - ***Note: "PUSH" is the process of saving onto the online repository, and thus is what would save your code if your computer were to crash and lose everything.***
    - "Pull-Request" - (Merging into main)
        - Once your development on the branch is complete, submit a "Pull-Request (PR)" so that the reviewers can check for conflicts and merge it into the main-branch.
        - Remember, the main-branch is the latest code that "works," thus the reviewing process makes sure that we will always have a production ready code even if the collaboration team grows to sizeable size.

## How to find documentation?
- The most recent version of the code has been updated with function signature and documentation in the code, thus there are three ways to find documentation on the code:
    - Method 1: Use `help({object})` and replace `{object}` with anything you need help with.
        - Ex: `help(lif.LIF_Network.simulate)` after we've ran `from src.model import lifNetwork as lif`
    - Method 2: Read the code
        - The source files are documented and with varaibles names to indicate their purposes. 
        - However, this is still a **work in progress** because of all the extra NOTE and TODO that are in there.
    - Method 3: Visit the documentation site - **WORK IN PROGRESS (NOT LIVE)**
