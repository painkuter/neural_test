package main

import (
	"database/sql"
	"fmt"
	"io/ioutil"

	_ "github.com/go-sql-driver/mysql"
)

const dbNname = "trading"

var connections = []string{
	"test",
	"root:111@tcp(127.0.0.1:3306)/",
	"root:12345678@tcp(127.0.0.1:3306)/",
	"root@tcp(127.0.0.1:3306)/",
	"root:111@localhost/",
}

func getMySQL() string {
	b, err := ioutil.ReadFile("dev/mtest_users.sql") // just pass the file name
	if err != nil {
		fmt.Println("ERROR Getting SQL file: " + err.Error())
	}
	return string(b) // convert content to a 'string'
}

func DB() *sql.DB {
	fmt.Println("Connecting to database")
	var (
		//dbConnect string
		db  *sql.DB
		err error
	)

	for _, el := range connections {
		db, err = sql.Open("mysql", el)
		result, _ := db.Exec("CREATE DATABASE IF NOT EXISTS " + dbNname)
		db, err = sql.Open("mysql", el+dbNname)
		if result != nil {
			fmt.Println("CONNECTED TO DB-SERVER: " + el)
			break
		}
	}
	if err != nil {
		fmt.Println("DB Initialization finished with errors:")
		panic(err)
	} else {
		fmt.Printf("DB Initialization finished successfully [%v]\n", dbNname)
	}
	// defer db.Close()
	return db
}
