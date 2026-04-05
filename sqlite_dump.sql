-- SQLite Database Dump

CREATE TABLE [Player] (
	[Player_Id]	integer NOT NULL,
	[Player_Name]	varchar(400) NOT NULL COLLATE NOCASE,
	[DOB]	datetime COLLATE NOCASE,
	[Batting_hand]	integer NOT NULL,
	[Bowling_skill]	integer,
	[Country_Name]	integer NOT NULL,
    PRIMARY KEY ([Player_Id])
,
    FOREIGN KEY ([Batting_hand])
        REFERENCES [Batting_Style]([Batting_Id]),
    FOREIGN KEY ([Bowling_skill])
        REFERENCES [Bowling_Style]([Bowling_Id]),
    FOREIGN KEY ([Country_Name])
        REFERENCES [Country]([Country_Id])
);

CREATE TABLE [Extra_Runs] (
	[Match_Id]	integer NOT NULL,
	[Over_Id]	integer NOT NULL,
	[Ball_Id]	integer NOT NULL,
	[Extra_Type_Id]	integer NOT NULL,
	[Extra_Runs]	integer NOT NULL,
	[Innings_No]	integer NOT NULL,
    PRIMARY KEY ([Match_Id], [Over_Id], [Ball_Id], [Innings_No])
,
    FOREIGN KEY ([Match_Id])
        REFERENCES [Ball_by_Ball]([Ball_Id]),
    FOREIGN KEY ([Over_Id])
        REFERENCES [Ball_by_Ball]([Ball_Id]),
    FOREIGN KEY ([Ball_Id])
        REFERENCES [Ball_by_Ball]([Ball_Id]),
    FOREIGN KEY ([Innings_No])
        REFERENCES [Ball_by_Ball]([Ball_Id]),
    FOREIGN KEY ([Match_Id])
        REFERENCES [Ball_by_Ball]([Innings_No]),
    FOREIGN KEY ([Over_Id])
        REFERENCES [Ball_by_Ball]([Innings_No]),
    FOREIGN KEY ([Ball_Id])
        REFERENCES [Ball_by_Ball]([Innings_No]),
    FOREIGN KEY ([Innings_No])
        REFERENCES [Ball_by_Ball]([Innings_No]),
    FOREIGN KEY ([Match_Id])
        REFERENCES [Ball_by_Ball]([Match_Id]),
    FOREIGN KEY ([Over_Id])
        REFERENCES [Ball_by_Ball]([Match_Id]),
    FOREIGN KEY ([Ball_Id])
        REFERENCES [Ball_by_Ball]([Match_Id]),
    FOREIGN KEY ([Innings_No])
        REFERENCES [Ball_by_Ball]([Match_Id]),
    FOREIGN KEY ([Match_Id])
        REFERENCES [Ball_by_Ball]([Over_Id]),
    FOREIGN KEY ([Over_Id])
        REFERENCES [Ball_by_Ball]([Over_Id]),
    FOREIGN KEY ([Ball_Id])
        REFERENCES [Ball_by_Ball]([Over_Id]),
    FOREIGN KEY ([Innings_No])
        REFERENCES [Ball_by_Ball]([Over_Id]),
    FOREIGN KEY ([Extra_Type_Id])
        REFERENCES [Extra_Type]([Extra_Id])
);

CREATE TABLE [Batsman_Scored] (
	[Match_Id]	integer NOT NULL,
	[Over_Id]	integer NOT NULL,
	[Ball_Id]	integer NOT NULL,
	[Runs_Scored]	integer NOT NULL,
	[Innings_No]	integer NOT NULL,
    PRIMARY KEY ([Match_Id], [Over_Id], [Ball_Id], [Innings_No])
,
    FOREIGN KEY ([Match_Id])
        REFERENCES [Ball_by_Ball]([Ball_Id]),
    FOREIGN KEY ([Over_Id])
        REFERENCES [Ball_by_Ball]([Ball_Id]),
    FOREIGN KEY ([Ball_Id])
        REFERENCES [Ball_by_Ball]([Ball_Id]),
    FOREIGN KEY ([Innings_No])
        REFERENCES [Ball_by_Ball]([Ball_Id]),
    FOREIGN KEY ([Match_Id])
        REFERENCES [Ball_by_Ball]([Innings_No]),
    FOREIGN KEY ([Over_Id])
        REFERENCES [Ball_by_Ball]([Innings_No]),
    FOREIGN KEY ([Ball_Id])
        REFERENCES [Ball_by_Ball]([Innings_No]),
    FOREIGN KEY ([Innings_No])
        REFERENCES [Ball_by_Ball]([Innings_No]),
    FOREIGN KEY ([Match_Id])
        REFERENCES [Ball_by_Ball]([Match_Id]),
    FOREIGN KEY ([Over_Id])
        REFERENCES [Ball_by_Ball]([Match_Id]),
    FOREIGN KEY ([Ball_Id])
        REFERENCES [Ball_by_Ball]([Match_Id]),
    FOREIGN KEY ([Innings_No])
        REFERENCES [Ball_by_Ball]([Match_Id]),
    FOREIGN KEY ([Match_Id])
        REFERENCES [Ball_by_Ball]([Over_Id]),
    FOREIGN KEY ([Over_Id])
        REFERENCES [Ball_by_Ball]([Over_Id]),
    FOREIGN KEY ([Ball_Id])
        REFERENCES [Ball_by_Ball]([Over_Id]),
    FOREIGN KEY ([Innings_No])
        REFERENCES [Ball_by_Ball]([Over_Id])
);

CREATE TABLE [Batting_Style] (
	[Batting_Id]	integer NOT NULL,
	[Batting_hand]	varchar(200) NOT NULL COLLATE NOCASE,
    PRIMARY KEY ([Batting_Id])

);

CREATE TABLE [Bowling_Style] (
	[Bowling_Id]	integer NOT NULL,
	[Bowling_skill]	varchar(200) NOT NULL COLLATE NOCASE,
    PRIMARY KEY ([Bowling_Id])

);

CREATE TABLE [Country] (
	[Country_Id]	integer NOT NULL,
	[Country_Name]	varchar(200) NOT NULL COLLATE NOCASE,
    PRIMARY KEY ([Country_Id])

);

CREATE TABLE [Season] (
	[Season_Id]	integer NOT NULL,
	[Man_of_the_Series]	integer NOT NULL,
	[Orange_Cap]	integer NOT NULL,
	[Purple_Cap]	integer NOT NULL,
	[Season_Year]	integer,
    PRIMARY KEY ([Season_Id])
,
    FOREIGN KEY ([Man_of_the_Series])
        REFERENCES [Player]([Player_Id]),
    FOREIGN KEY ([Orange_Cap])
        REFERENCES [Player]([Player_Id]),
    FOREIGN KEY ([Purple_Cap])
        REFERENCES [Player]([Player_Id])
);

CREATE TABLE [City] (
	[City_Id]	integer NOT NULL,
	[City_Name]	varchar(200) NOT NULL COLLATE NOCASE,
	[Country_id]	integer,
    PRIMARY KEY ([City_Id])
,
    FOREIGN KEY ([Country_id])
        REFERENCES [Country]([Country_Id])
);

CREATE TABLE [Outcome] (
	[Outcome_Id]	integer NOT NULL,
	[Outcome_Type]	varchar(200) NOT NULL COLLATE NOCASE,
    PRIMARY KEY ([Outcome_Id])

);

CREATE TABLE [Win_By] (
	[Win_Id]	integer NOT NULL,
	[Win_Type]	varchar(200) NOT NULL COLLATE NOCASE,
    PRIMARY KEY ([Win_Id])

);

CREATE TABLE [Wicket_Taken] (
	[Match_Id]	integer NOT NULL,
	[Over_Id]	integer NOT NULL,
	[Ball_Id]	integer NOT NULL,
	[Player_Out]	integer NOT NULL,
	[Kind_Out]	integer NOT NULL,
	[Fielders]	integer,
	[Innings_No]	integer NOT NULL,
    PRIMARY KEY ([Match_Id], [Over_Id], [Ball_Id], [Innings_No])
,
    FOREIGN KEY ([Match_Id])
        REFERENCES [Ball_by_Ball]([Ball_Id]),
    FOREIGN KEY ([Over_Id])
        REFERENCES [Ball_by_Ball]([Ball_Id]),
    FOREIGN KEY ([Ball_Id])
        REFERENCES [Ball_by_Ball]([Ball_Id]),
    FOREIGN KEY ([Innings_No])
        REFERENCES [Ball_by_Ball]([Ball_Id]),
    FOREIGN KEY ([Match_Id])
        REFERENCES [Ball_by_Ball]([Innings_No]),
    FOREIGN KEY ([Over_Id])
        REFERENCES [Ball_by_Ball]([Innings_No]),
    FOREIGN KEY ([Ball_Id])
        REFERENCES [Ball_by_Ball]([Innings_No]),
    FOREIGN KEY ([Innings_No])
        REFERENCES [Ball_by_Ball]([Innings_No]),
    FOREIGN KEY ([Match_Id])
        REFERENCES [Ball_by_Ball]([Match_Id]),
    FOREIGN KEY ([Over_Id])
        REFERENCES [Ball_by_Ball]([Match_Id]),
    FOREIGN KEY ([Ball_Id])
        REFERENCES [Ball_by_Ball]([Match_Id]),
    FOREIGN KEY ([Innings_No])
        REFERENCES [Ball_by_Ball]([Match_Id]),
    FOREIGN KEY ([Match_Id])
        REFERENCES [Ball_by_Ball]([Over_Id]),
    FOREIGN KEY ([Over_Id])
        REFERENCES [Ball_by_Ball]([Over_Id]),
    FOREIGN KEY ([Ball_Id])
        REFERENCES [Ball_by_Ball]([Over_Id]),
    FOREIGN KEY ([Innings_No])
        REFERENCES [Ball_by_Ball]([Over_Id]),
    FOREIGN KEY ([Kind_Out])
        REFERENCES [Out_Type]([Out_Id]),
    FOREIGN KEY ([Player_Out])
        REFERENCES [Player]([Player_Id]),
    FOREIGN KEY ([Fielders])
        REFERENCES [Player]([Player_Id])
);

CREATE TABLE [Venue] (
	[Venue_Id]	integer NOT NULL,
	[Venue_Name]	varchar(450) NOT NULL COLLATE NOCASE,
	[City_Id]	integer,
    PRIMARY KEY ([Venue_Id])
,
    FOREIGN KEY ([City_Id])
        REFERENCES [City]([City_Id])
);

CREATE TABLE [Extra_Type] (
	[Extra_Id]	integer NOT NULL,
	[Extra_Name]	varchar(150) NOT NULL COLLATE NOCASE,
    PRIMARY KEY ([Extra_Id])

);

CREATE TABLE [Out_Type] (
	[Out_Id]	integer NOT NULL,
	[Out_Name]	varchar(250) NOT NULL COLLATE NOCASE,
    PRIMARY KEY ([Out_Id])

);

CREATE TABLE [Toss_Decision] (
	[Toss_Id]	integer NOT NULL,
	[Toss_Name]	varchar(50) NOT NULL COLLATE NOCASE,
    PRIMARY KEY ([Toss_Id])

);

CREATE TABLE [Umpire] (
	[Umpire_Id]	integer NOT NULL,
	[Umpire_Name]	varchar(350) NOT NULL COLLATE NOCASE,
	[Umpire_Country]	integer NOT NULL,
    PRIMARY KEY ([Umpire_Id])
,
    FOREIGN KEY ([Umpire_Country])
        REFERENCES [Country]([Country_Id])
);

CREATE TABLE [Team] (
	[Team_Id]	integer NOT NULL,
	[Team_Name]	varchar(450) NOT NULL COLLATE NOCASE,
    PRIMARY KEY ([Team_Id])

);

CREATE TABLE [Ball_by_Ball] (
	[Match_Id]	integer NOT NULL,
	[Over_Id]	integer NOT NULL,
	[Ball_Id]	integer NOT NULL,
	[Innings_No]	integer NOT NULL,
	[Team_Batting]	integer NOT NULL,
	[Team_Bowling]	integer NOT NULL,
	[Striker_Batting_Position]	integer NOT NULL,
	[Striker]	integer NOT NULL,
	[Non_Striker]	integer NOT NULL,
	[Bowler]	integer NOT NULL,
    PRIMARY KEY ([Match_Id], [Over_Id], [Ball_Id], [Innings_No])
,
    FOREIGN KEY ([Match_Id])
        REFERENCES [Match]([Match_Id]),
    FOREIGN KEY ([Striker])
        REFERENCES [Player]([Player_Id]),
    FOREIGN KEY ([Non_Striker])
        REFERENCES [Player]([Player_Id]),
    FOREIGN KEY ([Bowler])
        REFERENCES [Player]([Player_Id]),
    FOREIGN KEY ([Team_Batting])
        REFERENCES [Team]([Team_Id]),
    FOREIGN KEY ([Team_Bowling])
        REFERENCES [Team]([Team_Id])
);

CREATE TABLE [sysdiagrams] (
	[name]	nvarchar(128) NOT NULL COLLATE NOCASE,
	[principal_id]	integer NOT NULL,
	[diagram_id]	integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	[version]	integer,
	[definition]	blob

);

CREATE TABLE [Match] (
	[Match_Id]	integer NOT NULL,
	[Team_1]	integer NOT NULL,
	[Team_2]	integer NOT NULL,
	[Match_Date]	datetime NOT NULL COLLATE NOCASE,
	[Season_Id]	integer NOT NULL,
	[Venue_Id]	integer NOT NULL,
	[Toss_Winner]	integer NOT NULL,
	[Toss_Decide]	integer NOT NULL,
	[Win_Type]	integer NOT NULL,
	[Win_Margin]	integer,
	[Outcome_type]	integer NOT NULL,
	[Match_Winner]	integer,
	[Man_of_the_Match]	integer,
    PRIMARY KEY ([Match_Id])
,
    FOREIGN KEY ([Outcome_type])
        REFERENCES [Outcome]([Outcome_Id]),
    FOREIGN KEY ([Man_of_the_Match])
        REFERENCES [Player]([Player_Id]),
    FOREIGN KEY ([Season_Id])
        REFERENCES [Season]([Season_Id]),
    FOREIGN KEY ([Team_1])
        REFERENCES [Team]([Team_Id]),
    FOREIGN KEY ([Team_2])
        REFERENCES [Team]([Team_Id]),
    FOREIGN KEY ([Toss_Winner])
        REFERENCES [Team]([Team_Id]),
    FOREIGN KEY ([Match_Winner])
        REFERENCES [Team]([Team_Id]),
    FOREIGN KEY ([Toss_Decide])
        REFERENCES [Toss_Decision]([Toss_Id]),
    FOREIGN KEY ([Venue_Id])
        REFERENCES [Venue]([Venue_Id]),
    FOREIGN KEY ([Win_Type])
        REFERENCES [Win_By]([Win_Id])
);

CREATE TABLE [Rolee] (
	[Role_Id]	integer NOT NULL,
	[Role_Desc]	varchar(150) NOT NULL COLLATE NOCASE,
    PRIMARY KEY ([Role_Id])

);

CREATE TABLE [Player_Match] (
	[Match_Id]	integer NOT NULL,
	[Player_Id]	integer NOT NULL,
	[Role_Id]	integer NOT NULL,
	[Team_Id]	integer,
    PRIMARY KEY ([Match_Id], [Player_Id])
,
    FOREIGN KEY ([Match_Id])
        REFERENCES [Match]([Match_Id]),
    FOREIGN KEY ([Player_Id])
        REFERENCES [Player]([Player_Id]),
    FOREIGN KEY ([Role_Id])
        REFERENCES [Rolee]([Role_Id]),
    FOREIGN KEY ([Team_Id])
        REFERENCES [Team]([Team_Id])
);

