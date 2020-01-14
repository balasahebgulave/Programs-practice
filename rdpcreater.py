import pandas as pd
data = pd.read_csv('machinedata.csv')




def updateRDPFile(teamnamelist,teamiplist):
	exist_file = open('YNS.rdg','r')
	exist_file_data = exist_file.readlines()
	exist_file_data = [i.strip() for i in exist_file_data]
	exist_file.close()
	print('------------exist_file_data-------------',exist_file_data)
	for teamname, teamip in zip(teamnamelist,teamiplist):
		new_data = ''
		for i in range(21):
			user = f"{teamname.lower()[0]}user00{i}"
		
			if i == 0:
				groupstart = f"""
					<group>
				      <properties>
				        <expanded>False</expanded>
				        <name>team{teamname}</name>
				      </properties>
					"""
				new_data+=groupstart
				team = f"team{teamname}-administrator"
				user = 'administrator'
			else:
				team = f"team{teamname}_{user}"
			
			filedata = f"""
				<server>
				      <properties>
				        <displayName>{team}</displayName>
				        <name>{teamip}</name>
				      </properties>
				      <logonCredentials inherit="None">
				        <profileName scope="Local">Custom</profileName>
				        <userName>{user}</userName>
				        <password></password>
				        <domain />
				      </logonCredentials>
				</server>
				"""
			new_data+=filedata
		groupend = "</group>"
		new_data+=groupend

		trigger = len(exist_file_data) - 6
		teamname = f'<name>team{teamname}</name>'
		print('------teamname--------',teamname)
		if teamname not in exist_file_data:
			file = open('YNS.rdg','w')
			updatefile = ''
			for index, row in enumerate(exist_file_data):
				if index == 7:
					row+=new_data	
					print('---------row--------',row)
					updatefile+=row
					

				else:
					updatefile+=row
			file.write(updatefile)
			file.close()

	return True


# updateRDPFile(teamnamelist,teamiplist)



def createRDPFile(teamnamelist,teamiplist):
	file = open('YNS.rdg','a')
	filestart = """<?xml version="1.0" encoding="utf-8"?>
			<RDCMan programVersion="2.7" schemaVersion="3">
			<file>
			<credentialsProfiles />
			<properties>
				<expanded>False</expanded>
				<name>YNS</name>
			</properties>
			"""
	file.writelines(filestart)

	for teamname, teamip in zip(teamnamelist,teamiplist):
		
		for i in range(21):
			user = f"{teamname.lower()[0]}user00{i}"
		
			if i == 0:
				groupstart = f"""
					<group>
				      <properties>
				        <expanded>False</expanded>
				        <name>team{teamname}</name>
				      </properties>
					"""
				file.writelines(groupstart)
				team = f"team{teamname}-administrator"
				user = 'administrator'
			else:
				team = f"team{teamname}_{user}"
			
			filedata = f"""
				<server>
				      <properties>
				        <displayName>{team}</displayName>
				        <name>{teamip}</name>
				      </properties>
				      <logonCredentials inherit="None">
				        <profileName scope="Local">Custom</profileName>
				        <userName>{user}</userName>
				        <password></password>
				        <domain />
				      </logonCredentials>
				</server>
				"""
			file.writelines(filedata)
		groupend = "</group>"
		file.writelines(groupend)

	fileend = """
			</file>
			<connected />
			<favorites />
			<recentlyUsed />
			</RDCMan>
			"""
	file.writelines(fileend)
	file.close()
	return True


teamnamelist = list(data['machine'])
teamiplist = list(data['machineip'])


def controller():
	
	print('----------teamiplist,teamnamelist----------',len(teamiplist),len(teamnamelist))

	try:
		updateRDPFile(teamnamelist,teamiplist)
	except Exception as e:
		print('---------error------------: ',e)
		createRDPFile(teamnamelist,teamiplist)



controller()
